import warnings, pickle, torch, math, os, random, numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import multitask_transformer_class

warnings.filterwarnings("ignore")


def initialize_training(prop):
    model = multitask_transformer_class.MultitaskTransformerModel(prop['task_type'], prop['device'], prop['nclasses'],
                                                                  prop['seq_len'], prop['batch'], \
                                                                  prop['input_size'], prop['emb_size'], prop['nhead'],
                                                                  prop['nhid'], prop['nhid_tar'], prop['nhid_task'],
                                                                  prop['nlayers'], prop['dropout']).to(prop['device'])
    best_model = multitask_transformer_class.MultitaskTransformerModel(prop['task_type'], prop['device'],
                                                                       prop['nclasses'], prop['seq_len'], prop['batch'], \
                                                                       prop['input_size'], prop['emb_size'],
                                                                       prop['nhead'], prop['nhid'], prop['nhid_tar'],
                                                                       prop['nhid_task'], prop['nlayers'],
                                                                       prop['dropout']).to(prop['device'])

    criterion_tar = torch.nn.MSELoss()
    criterion_task = torch.nn.CrossEntropyLoss() if prop[
                                                        'task_type'] == 'classification' else torch.nn.MSELoss()  # nn.L1Loss() for MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=prop['lr'])
    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=prop['lr'])  # get new optimiser

    return model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer


def attention_sampled_masking_heuristic(X, masking_ratio, ratio_highest_attention, instance_weights):
    # attention_weights = attention_weights.to('cpu')
    # instance_weights = torch.sum(attention_weights, axis = 1)
    res, index = instance_weights.topk(int(math.ceil(ratio_highest_attention * X.shape[1])))
    index = index.cpu().data.tolist()
    index2 = [random.sample(index[i], int(math.ceil(masking_ratio * X.shape[1]))) for i in range(X.shape[0])]
    return np.array(index2)


def random_instance_masking(X, masking_ratio, ratio_highest_attention, instance_weights):
    indices = attention_sampled_masking_heuristic(X, masking_ratio, ratio_highest_attention, instance_weights)
    boolean_indices = np.array([[True if i in index else False for i in range(X.shape[1])] for index in indices])
    boolean_indices_masked = np.repeat(boolean_indices[:, :, np.newaxis], X.shape[2], axis=2)
    boolean_indices_unmasked = np.invert(boolean_indices_masked)

    X_train_tar, y_train_tar_masked, y_train_tar_unmasked = np.copy(X), np.copy(X), np.copy(X)
    X_train_tar = np.where(boolean_indices_unmasked, X, 0.0)
    y_train_tar_masked = y_train_tar_masked[boolean_indices_masked].reshape(X.shape[0], -1)
    y_train_tar_unmasked = y_train_tar_unmasked[boolean_indices_unmasked].reshape(X.shape[0], -1)
    X_train_tar, y_train_tar_masked, y_train_tar_unmasked = torch.as_tensor(X_train_tar).float(), torch.as_tensor(
        y_train_tar_masked).float(), torch.as_tensor(y_train_tar_unmasked).float()

    return X_train_tar, y_train_tar_masked, y_train_tar_unmasked, boolean_indices_masked, boolean_indices_unmasked


def evaluate(y_pred, y, nclasses, criterion, task_type, device, avg):
    results = []

    if task_type == 'classification':
        loss = criterion(y_pred.view(-1, nclasses), torch.as_tensor(y, device=device)).item()

        pred, target = y_pred.cpu().data.numpy(), y.cpu().data.numpy()
        pred = np.argmax(pred, axis=1)
        acc = accuracy_score(target, pred)
        prec = precision_score(target, pred, average=avg)
        rec = recall_score(target, pred, average=avg)
        f1 = f1_score(target, pred, average=avg)

        results.extend([loss, acc, prec, rec, f1])
    else:
        y_pred = y_pred.squeeze()
        y = torch.as_tensor(y, device=device)
        rmse = math.sqrt(((y_pred - y) * (y_pred - y)).sum().data / y_pred.shape[0])
        mae = (torch.abs(y_pred - y).sum().data / y_pred.shape[0]).item()
        results.extend([rmse, mae])
    # per_class_results = precision_recall_fscore_support(target, pred, average = None, labels = list(range(0, nclasses)))

    return results


def test(model, X, y, batch, nclasses, criterion, task_type, device, avg):
    model.eval()  # Turn on the evaluation mode
    num_batches = math.ceil(X.shape[0] / batch)

    output_arr = []
    with torch.no_grad():
        for i in range(num_batches):
            start = int(i * batch)
            end = int((i + 1) * batch)
            num_inst = y[start: end].shape[0]

            out = model(torch.as_tensor(X[start: end], device=device), task_type)[0]
            output_arr.append(out[: num_inst])

    return evaluate(torch.cat(output_arr, 0), y, nclasses, criterion, task_type, device, avg)


def compute_tar_loss(model, device, criterion_tar, y_train_tar_masked, y_train_tar_unmasked, batched_input_tar, \
                     batched_boolean_indices_masked, batched_boolean_indices_unmasked, num_inst, start):
    model.train()
    out_tar = model(torch.as_tensor(batched_input_tar, device=device), 'reconstruction')[0]

    out_tar_masked = torch.as_tensor(
        out_tar[torch.as_tensor(batched_boolean_indices_masked)].reshape(out_tar.shape[0], -1), device=device)
    out_tar_unmasked = torch.as_tensor(
        out_tar[torch.as_tensor(batched_boolean_indices_unmasked)].reshape(out_tar.shape[0], -1), device=device)

    loss_tar_masked = criterion_tar(out_tar_masked[: num_inst],
                                    torch.as_tensor(y_train_tar_masked[start: start + num_inst], device=device))
    loss_tar_unmasked = criterion_tar(out_tar_unmasked[: num_inst],
                                      torch.as_tensor(y_train_tar_unmasked[start: start + num_inst], device=device))

    return loss_tar_masked, loss_tar_unmasked


def compute_task_loss(nclasses, model, device, criterion_task, y_train_task, batched_input_task, task_type, num_inst,
                      start):
    model.train()
    out_task, attn = model(torch.as_tensor(batched_input_task, device=device), task_type)
    out_task = out_task.view(-1, nclasses) if task_type == 'classification' else out_task.squeeze()
    loss_task = criterion_task(out_task[: num_inst], torch.as_tensor(y_train_task[start: start + num_inst],
                                                                     device=device))  # dtype = torch.long
    return attn, loss_task


def multitask_train(model, criterion_tar, criterion_task, optimizer, X_train_tar, X_train_task, y_train_tar_masked,
                    y_train_tar_unmasked, \
                    y_train_task, boolean_indices_masked, boolean_indices_unmasked, prop):
    model.train()  # Turn on the train mode
    total_loss_tar_masked, total_loss_tar_unmasked, total_loss_task = 0.0, 0.0, 0.0
    num_batches = math.ceil(X_train_tar.shape[0] / prop['batch'])
    output, attn_arr = [], []

    for i in range(num_batches):
        start = int(i * prop['batch'])
        end = int((i + 1) * prop['batch'])
        num_inst = y_train_task[start: end].shape[0]

        optimizer.zero_grad()

        batched_input_tar = X_train_tar[start: end]
        batched_input_task = X_train_task[start: end]
        batched_boolean_indices_masked = boolean_indices_masked[start: end]
        batched_boolean_indices_unmasked = boolean_indices_unmasked[start: end]

        loss_tar_masked, loss_tar_unmasked = compute_tar_loss(model, prop['device'], criterion_tar, y_train_tar_masked,
                                                              y_train_tar_unmasked, \
                                                              batched_input_tar, batched_boolean_indices_masked,
                                                              batched_boolean_indices_unmasked, num_inst, start)

        attn, loss_task = compute_task_loss(prop['nclasses'], model, prop['device'], criterion_task, y_train_task, \
                                            batched_input_task, prop['task_type'], num_inst, start)

        total_loss_tar_masked += loss_tar_masked.item()
        total_loss_tar_unmasked += loss_tar_unmasked.item()
        total_loss_task += loss_task.item() * num_inst

        # a = list(train_model.parameters())[0].clone()
        loss = prop['task_rate'] * (prop['lamb'] * loss_tar_masked + (1 - prop['lamb']) * loss_tar_unmasked) + (
                1 - prop['task_rate']) * loss_task
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # b = list(train_model.parameters())[0].clone()
        # print(torch.equal(a.data, b.data))

        # if list(model.parameters())[0].grad is None:
        #    print("None")

        # remove the diagonal values of the attention map while aggregating the column wise attention scores
        attn_arr.append(torch.sum(attn, axis=1) - torch.diagonal(attn, offset=0, dim1=1, dim2=2))

    instance_weights = torch.cat(attn_arr, axis=0)
    return total_loss_tar_masked, total_loss_tar_unmasked, total_loss_task / y_train_task.shape[0], instance_weights


def training(model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train_task, y_train_task,
             X_test, y_test, prop):
    tar_loss_masked_arr, tar_loss_unmasked_arr, tar_loss_arr, task_loss_arr, min_task_loss = [], [], [], [], math.inf
    acc, rmse, mae = 0, math.inf, math.inf

    instance_weights = torch.as_tensor(torch.rand(X_train_task.shape[0], prop['seq_len']), device=prop['device'])
    for epoch in range(1, prop['epochs'] + 1):

        X_train_tar, y_train_tar_masked, y_train_tar_unmasked, boolean_indices_masked, boolean_indices_unmasked = \
            random_instance_masking(X_train_task, prop['masking_ratio'], prop['ratio_highest_attention'],
                                    instance_weights)

        tar_loss_masked, tar_loss_unmasked, task_loss, instance_weights = multitask_train(model, criterion_tar,
                                                                                          criterion_task, optimizer,
                                                                                          X_train_tar, X_train_task,
                                                                                          y_train_tar_masked,
                                                                                          y_train_tar_unmasked,
                                                                                          y_train_task,
                                                                                          boolean_indices_masked,
                                                                                          boolean_indices_unmasked,
                                                                                          prop)

        tar_loss_masked_arr.append(tar_loss_masked)
        tar_loss_unmasked_arr.append(tar_loss_unmasked)
        tar_loss = tar_loss_masked + tar_loss_unmasked
        tar_loss_arr.append(tar_loss)
        task_loss_arr.append(task_loss)
        print('Epoch: ' + str(epoch) + ', TAR Loss: ' + str(tar_loss), ', TASK Loss: ' + str(task_loss))

        # save model and optimizer for lowest training loss on the end task
        if task_loss < min_task_loss:
            min_task_loss = task_loss
            best_model.load_state_dict(model.state_dict())
            best_optimizer.load_state_dict(optimizer.state_dict())

        # Saved best model state at the lowest training loss is evaluated on the official test set
        test_metrics = test(best_model, X_test, y_test, prop['batch'], prop['nclasses'], criterion_task,
                            prop['task_type'], prop['device'], prop['avg'])

        if prop['task_type'] == 'classification' and test_metrics[1] > acc:
            acc = test_metrics[1]
        elif prop['task_type'] == 'regression' and test_metrics[0] < rmse:
            rmse = test_metrics[0]
            mae = test_metrics[1]

    if prop['task_type'] == 'classification':
        print('Dataset: ' + prop['dataset'] + ', Acc: ' + str(acc))
    elif prop['task_type'] == 'regression':
        print('Dataset: ' + prop['dataset'] + ', RMSE: ' + str(rmse) + ', MAE: ' + str(mae))

    del model
    torch.cuda.empty_cache()
