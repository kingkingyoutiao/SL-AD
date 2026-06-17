import logging
from collections import OrderedDict
from sklearn.metrics import accuracy_score, auc, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
from backbone.CNN_Auto.CNN_Auto_fenlei import Fenlei_Model as CNN_Auto_Fenlei
from backbone.CNN_Auto.CNN_Auto import Model as CNN_Auto
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
import torch.nn as nn
from torch import optim
from data.SSL_dataset import *
from data.pca_fast_dataset import NPZ_PCA_Fenlei_Dataset_train,NPZ_PCA_Fenlei_Dataset_test
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
def save_data(data, path, name):
    df = pd.DataFrame(data)
    os.makedirs(path,exist_ok=True)
    df.to_csv(os.path.join(path, name))
def get_args():
    current_file = os.path.abspath(__file__)
    script_directory = os.path.dirname(current_file)
    parser = argparse.ArgumentParser(description='SL-AD')
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--model_id', type=str, default='CNN3_77_FT3_GLAFF', help='model id')
    parser.add_argument('--model', type=str, default='CNN_Auto',
                        help='model name, options: [Autoformer, Informer, Transformer, CNN_Auto]')
    parser.add_argument('--flag', type=str, default='Plugin',
                            choices=['Plugin', 'Standard'], help='GLAFF or Standard')
    # data loader
    parser.add_argument('--data', type=str, default='AETA_fenlei', help='dataset type')

    parser.add_argument('--root_path', type=str, default='/media/scw4750/liuxiuhan/jingxilong/Autoformer-main/dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='AETA_1(17-22)_300_updated.csv', help='EQ file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t',#h
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--hist_len', type=int, default=168, help='length of history window')
    parser.add_argument('--q', type=float, default=0.75, help='quantile')
    parser.add_argument('--dim', type=int, default=256, help='dimension of hidden state')
    parser.add_argument('--dff', type=int, default=512, help='dimension of feed forward')
    parser.add_argument('--head_num', type=int, default=8, help='number of heads')
    parser.add_argument('--layer_num', type=int, default=2, help='number of layers')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=168, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=42, help='start token length')
    parser.add_argument('--pred_len', type=int, default=168, help='prediction sequence length')
    parser.add_argument('--num_cnn_layers', type=int, default=3, help='CNN_num_cnn_layers')
    # model define
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=3, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', default=False, action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    #Model BiLSTM
    parser.add_argument('--hidden_nc', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')#10
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=3000, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=80*9, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=60, help='early stopping patience')#3
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    parser.add_argument('--encoderpath',type=str,default='',help='The savepath of encout')
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    return args


def _load_model_state(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=args.devices)
    state_dict = checkpoint
    if list(state_dict.keys())[0].startswith("module.") and not isinstance(model, torch.nn.DataParallel):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "", 1)
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys not loaded: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    return model
def main(args):

    results_path = './fenlei_results'
    os.makedirs(results_path, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.use_gpu = torch.cuda.is_available() and args.use_gpu

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(i) for i in device_ids]
        args.gpu = args.device_ids[0]

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    auto = CNN_Auto(args).float()

    if args.use_multi_gpu and args.use_gpu:
        auto = nn.DataParallel(auto, device_ids=args.device_ids)

    model = CNN_Auto_Fenlei([auto], args)

    checkpoint = torch.load('./checkpoint/results_prauc.pth')

    new_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        new_checkpoint[k.replace('module.', '')] = v

    model.load_state_dict(new_checkpoint)

    if args.use_multi_gpu and args.use_gpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    train_data = NPZ_PCA_Fenlei_Dataset_train()
    test_data = NPZ_PCA_Fenlei_Dataset_test()

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        drop_last=False,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_data,
        batch_size=800,
        shuffle=False,
        num_workers=args.num_workers
    )

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    lr_adjust = {
        50: 5e-6,
        90: 1e-6,
        100: 1e-4,
        120: 5e-7,
        150: 1e-7,
        180: 5e-8,
        200: 1e-8
    }

    criterion = nn.CrossEntropyLoss()

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adam(
        parameters,
        lr=args.learning_rate
    )

    best_auc = 0.0

    # inference only
    if args.is_training == 0:
        test_loss_list = []
        test_loss_list, true_label, predictions, model_output, memory_allocate, memory_reserved, memory_usage, test_time = test(
            test_loader, model, criterion, test_loss_list, args)
        accuracy, fp, fn, auc_score, pr_auc, precision_test, recall_test, fpr, fnr, f1 = evaluate(true_label,
                                                                                                  predictions,
                                                                                                  model_output)
        print('ACC:', accuracy, 'ROC_AUC:', auc_score, 'PR_AUC:', pr_auc, 'Best_AUC:', best_auc, 'RECALL:', recall_test,
              'PRECISION:', precision_test, 'FPR:', fpr, 'FNR:', fnr, 'F1-SCORE:', f1)
        return

    # training
    for epoch in range(args.train_epochs + 1):

        print('*' * 30)

        if epoch in lr_adjust:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_adjust[epoch]

            print(
                f"Epoch {epoch}: "
                f"Learning rate adjusted to {lr_adjust[epoch]}"
            )

        train_loss = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            args
        )

        if epoch < 1:
            continue

        test_loss_list, true_label, predictions, model_output, memory_allocate, memory_reserved, memory_usage, test_time = test(
            test_loader, model, criterion, test_loss_list, args)

        accuracy, fp, fn, auc_score, pr_auc, precision_test, \
        recall_test, fpr, fnr, f1 = evaluate(
            true_label,
            predictions,
            model_output
        )

        print(
            'ACC:', accuracy,
            'ROC_AUC:', auc_score,
            'PR_AUC:', pr_auc,
            'RECALL:', recall_test,
            'PRECISION:', precision_test,
            'FPR:', fpr,
            'FNR:', fnr,
            'F1-SCORE:', f1
        )

        if auc_score > best_auc:

            best_auc = auc_score

            print(
                f'Epoch {epoch} is better! '
                f'Saving model...'
            )

            torch.save(
                model.state_dict(),
                os.path.join(
                    results_path,
                    'results_prauc.pth'
                )
            )
def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()

    running_loss = 0.0

    for i, (batch_x, batch_x_mark, labels, _, _, p0s) in enumerate(train_loader):

        batch_x = batch_x.float()
        batch_x_mark = batch_x_mark.float()
        labels = labels.long()

        if args.gpu is not None:
            batch_x = batch_x.cuda(args.gpu)
            batch_x_mark = batch_x_mark.cuda(args.gpu)
            labels = labels.cuda(args.gpu)

        outputs, _ = model(batch_x, batch_x_mark)

        p0s = p0s.float().to(args.gpu)
        loss = criterion(outputs, labels, p0=p0s)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def test(test_loader, model, criterion, loss_list, args):
    model.eval()

    test_time_all = []
    true_label = []
    predictions = []
    model_output = []
    running_loss = 0.0

    with torch.no_grad():
        for i, (batch_x, batch_x_mark, labels, _, _, p0) in enumerate(test_loader):

            test_time_start = time.time()

            batch_x = batch_x.to(torch.float32)
            batch_x_mark = batch_x_mark.to(torch.float32)
            labels = labels.to(torch.long)

            if args.gpu is not None:
                batch_x = batch_x.cuda(args.gpu)
                batch_x_mark = batch_x_mark.cuda(args.gpu)
                labels = labels.cuda(args.gpu)

            outputs, _ = model(batch_x, batch_x_mark)

            test_time_end = time.time() - test_time_start
            test_time_all.append(test_time_end)

            p0s = p0.float().to(args.gpu)
            loss = criterion(outputs, labels, p0=p0s)
            running_loss += loss.item()

            if args.gpu is not None:
                memory_allocate = torch.cuda.memory_allocated(args.gpu)
                memory_reserved = torch.cuda.memory_reserved(args.gpu)
                memory_usage = memory_allocate + memory_reserved
            else:
                memory_allocate = 0
                memory_reserved = 0
                memory_usage = 0

            pred = torch.argmax(outputs, dim=1)

            true_label.append(labels.cpu().numpy())
            predictions.append(pred.cpu().numpy())
            model_output.append(outputs.cpu().numpy())

            if (i + 1) % 1000 == 0:
                print('[Iteration:%5d] loss: %.3f' %
                      (i + 1, running_loss / 1000))
                loss_list.append(running_loss / 1000)
                running_loss = 0.0

    true_label = np.concatenate(true_label, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    model_output = np.concatenate(model_output, axis=0)

    correct = torch.sum(
        torch.tensor(predictions) == torch.tensor(true_label)
    ).item()

    total = len(true_label)

    print('Accuracy: %.2f %%' % (100 * correct / total))

    test_time = np.mean(np.array(test_time_all))
    print('test time:{}'.format(test_time))

    return loss_list, true_label, predictions, model_output, memory_allocate, memory_reserved, memory_usage, test_time

def evaluate(label, pred, output):
    accuracy = accuracy_score(label, pred)
    fp = false_positive_rate(label, pred)
    fn = false_negative_rate(label, pred)
    fpr, tpr, thresholds = roc_curve(label, output[:, 1], pos_label=1)
    auc_score = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(label, output[:, 1])
    pr_auc = auc(recall, precision)
    precision = precision_score(label, pred)
    recall = recall_score(label, pred)
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    f1 = f1_score(label, pred)
    return accuracy, fp, fn, auc_score, pr_auc, precision, recall, fpr, fnr, f1

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    return fpr

def false_negative_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp)
    return fnr
if __name__ == '__main__':
    args=get_args()
    main(args)