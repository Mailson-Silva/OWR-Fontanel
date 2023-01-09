import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from loss import SNNLoss, MarginLoss
from deep_nno import DeepNNO
import copy
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as T
import random
import numpy.random as npr

def save_image(tensor, index):
    t = T.ToPILImage()
    pil_image = t(tensor)
    pil_image.save(f'{index}.png')

class Trainer:
    def __init__(self, opts, network, discriminator, device, logger, augment_ops=None):
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.bce = nn.BCELoss(reduction='mean')
        self.distill_loss = nn.MSELoss(reduction='mean')
        self.snnl = SNNLoss()
        self.margin_loss = MarginLoss()
        self.augment_ops = augment_ops

        # DA method
        self.rsda = opts.rsda
        self.self_challenging = opts.self_challenging

        # store network
        self.network = network
        self.discriminator = discriminator
        self.network2 = None

        # method section
        self.features_dw = opts.features_dw
        self.snnl_weight = opts.snnlw
        self.ce_weight = opts.ce
        self.bce_weight = opts.bce
        self.ss_weight = opts.ssw
        self.nno = opts.nno

        # tau
        self.deep_nno = opts.deep_nno  # deep nno
        self.tau_val = not opts.no_tau_val
        self.multiple_taus = opts.multiple_taus
        if opts.multiple_taus:
            self.tau = Parameter(torch.ones(opts.initial_classes, device=device)*0.5, requires_grad=True)
        else:
            self.tau = Parameter(torch.tensor([0.5], device=device), requires_grad=True)

        if self.deep_nno or self.nno:
            self.deep_nno_handler = DeepNNO(self.tau, device, factor=opts.tau_factor, bm=self.deep_nno)

        # others
        self.device = device
        self.logger = logger
        self.num_classes = opts.initial_classes
        self.epochs = opts.epochs
        self.dataset = opts.dataset

    def next_iteration(self, new_classes):
        self.network2 = copy.deepcopy(self.network)
        self.network2.eval()
        # Duplicate current network to distillate info
        #self.network.linear.reset()  # reset the counters for NCM
        # Prepare internal structure (allocate mean array, etc) for the new classes
        self.network.add_classes(new_classes)
        # Store the new number of classes
        self.num_classes += new_classes

        if self.tau_val and self.multiple_taus:
            self.tau = Parameter(torch.cat((self.tau, torch.ones(new_classes, device=self.device)*0.5), 0))
        if self.tau_val and not self.multiple_taus:
            self.tau = Parameter(torch.tensor([0.5], device=self.device), requires_grad=True)

    def reject(self, x, dist, tau=None):
        out = torch.zeros(x.shape[0], x.shape[1] + 1).to(x.device)

        if self.deep_nno or self.nno:
            out[:, :x.shape[1]] = (x > tau).float() * 1. * x
        else:
            out[:, :x.shape[1]] = (dist <= tau).float() * 1. * x

        # last column contains probabilities (distances) for unknown.
        out[:, -1] = 1. - ((out[:, :x.shape[1]]).sum(1) > 0).float()
        return out

    def train(self, epoch, train_loader, subset_trainloader, optimizer, class_dict, iteration):
        # Training, single epoch
        print(f'Epoch: {epoch}')

        if iteration == 0 or not self.nno:
            self.network.train()
        else:
            self.network.eval()

        train_loss = 0
        ss_loss=0
        correct = 0
        ss_correct = 0
        total = 0
        count = 0
        print(f"Tau: {self.tau}")


        for idx, (inputs, targets_prep) in enumerate(train_loader):
            count += 1

            # one_hot_encoding targets
            targets_prep = targets_prep.to(self.device)
            targets = torch.zeros(inputs.shape[0], len(class_dict.keys())).to(self.device)
            targets.scatter_(1, targets_prep.view(-1, 1), 1).view(inputs.shape[0], -1)

            inputs = inputs.to(self.device)
            optimizer.zero_grad()
            outputs, feat = self.network(inputs)

            #print(np.shape(outputs), np.shape(feat))


            prediction, exp, distances = self.network.predict(outputs)  # prediction from NCM
            #print(np.shape(prediction), type(prediction))
            loss_bx = self.ce_weight * self.ce(exp, targets_prep) + \
                      self.snnl_weight * self.snnl(outputs, targets_prep) + \
                      self.bce_weight * self.bce(prediction, targets)

            # add distillation to losses
            if iteration > 0 and self.features_dw > 0:
                outputs_old,_ = self.network2(inputs)
                outputs_old = outputs_old.detach()
                loss_bx = loss_bx + self.features_dw * (self.distill_loss(outputs, outputs_old))


            if iteration == 0 or not self.nno and not (epoch == 0 and idx == 0):
                loss_bx.backward() #calcula o gradiente para todos os parametros x a serem otimizados
                optimizer.step() #atualiza x com o gradiente obtido anteriormente
                train_loss += loss_bx.item()

            # ## UPDATE MEANS ###
            # For the just learned classes it computes the online (moving) mean
                self.network.update_means(outputs, targets_prep.to('cpu'))

            #if not self.deep_nno and not self.nno:
            #    self.tau.data = self.network.linear.get_average_dist(dim=-1)

            if self.deep_nno or (iteration == 0 and self.nno):
                # Update tau parameters
                self.deep_nno_handler.update_taus(prediction, targets_prep, self.num_classes)
                #testar outra função para calculo de tau já que bm = False



            # Display loss and accuracy
            _, predicted = prediction.max(1)
            #print(predicted)
            total += targets_prep.size(0)
            #print(targets_prep)
            correct += predicted.eq(targets_prep).sum().item()


            if count % 2 == 0: #2 == 0
                print(f'[{int((100. * idx) / len(train_loader)):03d}%] == Loss: {train_loss / count:.3f}, '
                      f'Acc: {100. * correct / total:.3f} [{correct}/{total}]')

        self.logger.log_training(epoch, train_loss / len(train_loader), 100. * correct / total, iteration)

        #if self.ss_weight:
        #    self.logger.log_ss(epoch, ss_loss / len(train_loader), 100. * ss_correct / total, iteration)

    def test_closed_world(self, test_loader, display=True):
        self.network.eval()
        correct = 0
        total = 0
        count = 0

        with torch.no_grad():
            for idx, (inputs, targets_prep) in enumerate(test_loader):
                #print('targets_prep:', targets_prep)
                count += 1
                inputs = inputs.to(self.device)
                outputs, _= self.network(inputs)
                outputs, _, distances = self.network.predict(outputs)
                targets_prep = targets_prep.to(outputs.device)
                _, predicted = outputs.max(1)
                total += inputs.size(0)
                correct += predicted.eq(targets_prep).sum().item()

        acc = 100. * correct / total
        if display:
            print(f"TEST ACCURACY: {acc}")
        return acc

    def test_open_set(self, test_loader, last_class):
        print(f"Final Tau: {self.tau.data.cpu().numpy()}")

        self.network.eval()
        correct = 0
        total = 0
        count = 0
        unk = 0.
        tp = 0.
        rejected = 0.
        total_rejected = 0.
        stat_distances = [0., 0., 0.]
        stat_pred = [0., 0., 0.]

        with torch.no_grad():
            for idx, (inputs, targets_prep) in enumerate(test_loader):
                count += 1

                inputs = inputs.to(self.device)
                outputs,_ = self.network(inputs)
                outputs, _, distances = self.network.predict(outputs)
                targets_prep = targets_prep.to(outputs.device)
                # changes the probabilities (distances) for known and unknown
                new_outputs = self.reject(outputs, distances, self.tau)
                _, predicted = new_outputs.max(1)
                #print(f'count: {count}, labels: {targets_prep}')
                #print(f'output: {outputs}  ,  new_output: {new_outputs}' )
                # last_class is unknown class
                unk += (targets_prep == last_class).sum().item()
                # true positive
                tp += ((targets_prep == last_class) * predicted.eq(targets_prep)).sum().item()
                rejected += (predicted == last_class).sum().item()

                total += inputs.size(0)
                correct += predicted.eq(targets_prep).sum().item()
                stat_distances[0] += distances.min(dim=1)[0].sum()
                stat_distances[1] += distances.max(dim=1)[0].sum()
                stat_distances[2] += distances.mean(dim=1).sum()
                stat_pred[0] += outputs.min(dim=1)[0].sum()
                stat_pred[1] += outputs.max(dim=1)[0].sum()
                stat_pred[2] += outputs.mean(dim=1).sum()
                total_rejected += new_outputs[predicted == last_class].sum()

        acc = 100. * correct / total

        if rejected == 0:
            precision = 0
        else:
            precision = 100. * tp / rejected
        if unk == 0:
            recall = 0.
        else:
            recall = 100. * tp / unk
        if (precision + recall) == 0:
            f1score = 0
        else:
            f1score = 2. * (precision * recall) / (precision + recall)

        print(f"Accuracy: {acc:.2f}; Rej Rate {rejected / total:.2f}, Avg Pred Rej {total_rejected / total:.2f}"
              f"\n\t Min Pred {stat_pred[0] / total:.2f}, Max Pred {stat_pred[1] / total:.2f}, Mean_Pred {stat_pred[2] / total:.2f}"
              f"\n\t Min Dist {stat_distances[0] / total:.2f}, Max Dist {stat_distances[1] / total:.2f}, Mean Dist {stat_distances[2] / total:.2f}")

        return acc, precision, recall, f1score, rejected, unk

    def test_OWR(self, test_loader, last_class):
        print(f"Final Tau: {self.tau.data.cpu().numpy()}")

        self.network.eval()
        correct = 0
        total = 0
        count = 0
        unk = 0.
        tp = 0.
        rejected = 0.
        total_rejected = 0.
        stat_distances = [0., 0., 0.]
        stat_pred = [0., 0., 0.]
        sel_objs = []
        score_objs = []

        with torch.no_grad():
            for idx, (inputs, targets_prep) in enumerate(test_loader):
                count += 1

                inputs = inputs.to(self.device)
                outputs,_ = self.network(inputs)
                outputs, _, distances = self.network.predict(outputs)
                targets_prep = targets_prep.to(outputs.device)
                # changes the probabilities (distances) for known and unknown

                new_outputs = self.reject(outputs, distances, self.tau)
                #print(new_outputs)
                _, predicted = new_outputs.max(1)
                #print(f'count: {count}, labels: {targets_prep}')
                #print(f'output: {outputs}  ,  new_output: {new_outputs}' )
                # last_class is unknown class
                #unk += (targets_prep == last_class).sum().item()
                #print(new_outputs, type(new_outputs))
                if idx == 0:
                    indices = np.where(predicted == last_class)[0]

                    sel_objs.extend(indices)
                    new_outputs_copy = new_outputs.detach().cpu().numpy()
                    score_objs.extend(new_outputs_copy[indices, last_class])
                    '''
                    indices = [x for x in indices if targets_prep[x] == last_class]
                    if indices != [None] and indices!= []:
                        sel_objs.extend(indices)
                        new_outputs_copy = new_outputs.detach().cpu().numpy()
                        score_objs.extend(new_outputs_copy[indices,last_class])
                    '''

                else:
                    indices = np.where(predicted == last_class)[0]

                    new_outputs_copy = new_outputs.detach().cpu().numpy()
                    sel_objs.extend(indices + 15 * idx)
                    score_objs.extend(new_outputs_copy[indices, last_class])
                    '''
                    indices = [x for x in indices if targets_prep[x] == last_class]
                    if indices != [None] and indices!= []:
                        indices = np.array(indices)
                        new_outputs_copy = new_outputs.detach().cpu().numpy()
                        sel_objs.extend(indices + 15*idx)
                        score_objs.extend(new_outputs_copy[indices,last_class])
                    '''


                rejected += (predicted == last_class).sum().item()

                total += inputs.size(0)


                if last_class in targets_prep:

                    np_vr = targets_prep.detach().cpu().numpy()
                    id = np.where(np_vr == last_class)
                    np_vr[id] = 1E10
                    targets_prep = torch.from_numpy(np_vr)
                    #print(targets_prep)
                    correct += predicted.eq(targets_prep).sum().item()
                else:

                    correct += predicted.eq(targets_prep).sum().item()


                total_rejected += new_outputs[predicted == last_class].sum()

            #print(sel_objs, len(sel_objs))
            #print(score_objs, len(score_objs))

            sel_objs = np.array(sel_objs)
            score_objs = np.array(score_objs)

            sel = score_objs.argsort()[-5:][::-1]

            sel_objs = sel_objs[sel]
            print(sel_objs)


        acc = 100. * correct / total
        print(f'rejected: {rejected}')
        print(f"Accuracy: {acc:.2f}; Rej Rate {rejected / total:.2f}, Avg Pred Rej {total_rejected / total:.2f}")

        return acc, rejected, sel_objs

    def valid(self, epoch, valid_loader, optimizer, iteration, class_dict):
        self.network.eval()

        valid_loss = 0

        print(f"Tau: {self.tau.mean().cpu().detach().numpy()}")

        for idx, (inputs, targets_prep) in enumerate(valid_loader):

            optimizer.zero_grad()

            inputs = inputs.to(self.device)
            outputs,_ = self.network(inputs)
            predictions, _, distances = self.network.predict(outputs)  # prediction from NCM, not FC
            _, predicted = predictions.max(1)
            targets_prep = targets_prep.to(outputs.device)
            # one_hot_encoding targets
            targets = torch.zeros(inputs.shape[0], len(class_dict.keys())).to(self.device)
            targets.scatter_(1, targets_prep.view(-1, 1), 1).view(inputs.shape[0], -1)

            loss_bx = self.margin_loss(distances[predicted == targets_prep],
                                       targets[predicted == targets_prep], self.tau)

            if iteration == 0 or not self.nno:
                loss_bx.backward()
                optimizer.step()
                valid_loss += loss_bx.item()

        self.logger.log_valid(epoch, valid_loss / len(valid_loader), self.tau.mean().item(), iteration)

    def state_dict(self):
        state = {"tau": self.tau}
        return state

    def load_state_dict(self, state):
        if state["tau"] is not None:
            self.tau = Parameter(torch.tensor(state["tau"], device=self.device), requires_grad=True)
