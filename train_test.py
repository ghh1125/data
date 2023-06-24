""" Training and testing of the model
"""
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from model import MMDynamic
from dataloader import get_train_data_loader, get_test_data_loader
cuda = True if torch.cuda.is_available() else False



def train_epoch(data_list, label, model, optimizer):
    model.train()
    optimizer.zero_grad()
    MMLoss, LCA_Loss, MMlogit = model(data_list, label)
    MMLoss = torch.mean(MMLoss)
    # LCA_Loss = torch.mean(LCA_Loss)
    MMLoss.backward()
    optimizer.step()

    return MMLoss, LCA_Loss, MMlogit


def test_epoch(data_list, labels_te, model):
    model.eval()
    with torch.no_grad():
        logit = model.infer(data_list, labels_te)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
    return prob




#start
def train(datafolder):
    test_inverval = 20
    if 'BRCA' in datafolder:
        hidden_dim = [800]
        num_epoch = 2501
        lr = 0.0001
        step_size = 500
        num_class = 5
        dim_list = [1000, 1000, 503]
        seq_len = 500
        bs = 700
        test_inverval = 50
        cnt = 612
    elif 'TP' in datafolder:
        hidden_dim = [768]
        num_epoch = 2001
        lr = 0.0001
        step_size = 500
        num_class = 3
        dim_list = [768, 768]
        seq_len = 768
        bs = 64
        test_inverval = 10
        cnt = 1907
    elif 'ROSMAP' in datafolder:
        hidden_dim = [300]
        num_epoch = 1501
        lr = 0.0001
        step_size = 500
        num_class = 2
        dim_list = [200, 200, 200]
        seq_len = 200
        bs = 500
        test_inverval = 50
        cnt = 245

    train_data_loader = get_train_data_loader(data_folder=datafolder, batch_size=bs)
    test_data_loader = get_test_data_loader(data_folder=datafolder, batch_size=bs)



    model = MMDynamic(dim_list, hidden_dim, num_class, dropout=0.5)  # model
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)

    best_acc = -1
    best_F1_weighted = -1
    best_F1_macro = -1
    print("\nTraining...")
    for epoch in range(num_epoch):
        sum_loss = 0
        for idx, (labels_tr, data_train_list) in enumerate(train_data_loader):
            labels_tr = labels_tr.cuda()

            data_train_list = torch.stack(data_train_list)
            data_train_list = data_train_list.view(len(data_train_list), len(data_train_list[0]), seq_len, 1)
            data_train_list= data_train_list.cuda()

            loss, _, tr_prob = train_epoch(data_train_list, labels_tr, model, optimizer)
            sum_loss = sum_loss + loss.item()



        if epoch % test_inverval == 0:
            print("\nTraining: Epoch {:d}".format(epoch))
            print("loss: {:.5f}".format(sum_loss/cnt))
            # print("Training ACC: {:.5f}".format(accuracy_score(labels_tr.cpu(), tr_prob.argmax(1))))




        scheduler.step()
        for idx, (labels_te, data_test_list) in enumerate(test_data_loader):
            data_test_list = torch.stack(data_test_list)
            data_test_list = data_test_list.view(len(data_test_list), len(data_test_list[0]), seq_len, 1)
            data_test_list = data_test_list.cuda()




        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_test_list, labels_te, model)
            print("\nTest: Epoch {:d}".format(epoch))
            print("Test ACC: {:.5f}".format(accuracy_score(labels_te, te_prob.argmax(1))))
            if accuracy_score(labels_te, te_prob.argmax(1))>best_acc:
                best_acc = accuracy_score(labels_te, te_prob.argmax(1))
            if 'ROSMAP' in datafolder:
                print("Test AUC: {:.5f}".format(roc_auc_score(labels_te, te_prob[:, 1])))
                print("Test F1: {:.5f}".format(f1_score(labels_te, te_prob.argmax(1))))
            else :
                print("Test F1 weighted: {:.5f}".format(f1_score(labels_te, te_prob.argmax(1), average='weighted')))
                if f1_score(labels_te, te_prob.argmax(1), average='weighted')>best_F1_weighted:
                    best_F1_weighted = f1_score(labels_te, te_prob.argmax(1), average='weighted')
                print("Test F1 macro: {:.5f}".format(f1_score(labels_te, te_prob.argmax(1), average='macro')))
                if f1_score(labels_te, te_prob.argmax(1), average='macro')>best_F1_macro:
                    best_F1_macro = f1_score(labels_te, te_prob.argmax(1), average='macro')
        print('best_acc',best_acc,'best_F1_weighted',best_F1_weighted,'best_F1_macro',best_F1_macro)





