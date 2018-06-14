import time

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as functional

from model import LeNet, to_var, build_model


def train_lre(hyperparameters, data_loader, val_data, val_labels, test_loader, n_val=5):
    net, opt = build_model(hyperparameters)

    plot_step = 10

    accuracy_log = dict()
    data_log = dict()
    subselect_val = n_val * val_labels.unique().size()[0] != len(val_labels)
    meta_net = LeNet(n_out=1)
    for i in range(hyperparameters['num_iterations']):
        t = time.time()
        net.train()
        # Line 2 get batch of data
        image, labels, index = next(iter(data_loader))
        # since validation data is small I just fixed them instead of building an iterator
        # initialize a dummy network for the meta learning of the weights
        meta_net.load_state_dict(net.state_dict())

        if torch.cuda.is_available():
            meta_net.cuda()

        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        y_f_hat = meta_net(image)  # Line 4
        cost = functional.binary_cross_entropy_with_logits(y_f_hat, labels, reduce=False)  # Line 5
        eps = to_var(torch.zeros(cost.size()))  # Line 5
        l_f_meta = torch.sum(cost * eps)

        meta_net.zero_grad()

        # Line 6 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
        meta_net.update_params(hyperparameters['lr'], source_params=grads)

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        if subselect_val:
            class_inds = list()
            for c in val_labels.unique():
                matching_inds = (val_labels == c).nonzero()
                class_inds.append(matching_inds[np.random.permutation(len(matching_inds))[:n_val]])
            class_inds = torch.cat(class_inds)
            val_input = val_data[class_inds].squeeze_(1)
            val_output = val_labels[class_inds].squeeze_(1)
        else:
            val_input = val_data
            val_output = val_labels

        y_g_hat = meta_net(val_input)

        l_g_meta = functional.binary_cross_entropy_with_logits(y_g_hat, val_output)

        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f_hat = net(image)
        cost = functional.binary_cross_entropy_with_logits(y_f_hat, labels, reduce=False)
        data_log[i] = pd.DataFrame(
            data={
                'w': w.numpy(),
                'grad_eps': grad_eps.numpy(),
                'pre_w_cost': cost.detach().numpy(),
                'label': labels.numpy(),
                'index': index.numpy(),
            })
        l_f = torch.sum(cost * w)

        # print(data_log[i].head())
        # print(data_log[i].tail())

        opt.zero_grad()
        l_f.backward()
        opt.step()

        # meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item()
        # meta_losses_clean.append(meta_l / (1 - smoothing_alpha ** (i + 1)))
        #
        # net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item()
        # net_losses.append(net_l / (1 - smoothing_alpha ** (i + 1)))
        #
        print(time.time() - t)
        if i % plot_step == 0:
            print(i)
            net.eval()

            acc_df = {'preds': [], 'labels': [], 'index': []}
            for itr, (test_img, test_label, test_idx) in enumerate(test_loader):
                test_img = to_var(test_img, requires_grad=False)
                test_label = to_var(test_label, requires_grad=False)

                output = net(test_img)
                predicted = functional.sigmoid(output)

                acc_df['preds'].extend(predicted.detach().numpy().tolist())
                acc_df['labels'].extend(test_label.numpy().tolist())
                acc_df['index'].extend(test_idx.numpy().tolist())

            accuracy_log[i] = pd.DataFrame(acc_df).sort_values(['labels', 'index']).set_index('index', drop=True)


    data_log = pd.concat(data_log)
    data_log.index.set_names('iteration', level=0, inplace=True)
    accuracy_log = pd.concat(accuracy_log)
    accuracy_log.index.set_names('iteration', level=0, inplace=True)
    return data_log, accuracy_log
