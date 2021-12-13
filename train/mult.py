import os
import sys
import torch
import torch.nn as nn
import torch.optim as opt
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.eval_metrics import *

def train_mult(args, model, train_loader, valid_loader, test_loader):

    ### set basic staff 
    if args.use_gpu:
        model = model.cuda()
    optimizer = getattr(opt, args.optimizer)(model.parameters(), lr=args.lr)
    criterion = getattr(nn, args.criterion)()

    def train():
        model.train()
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            
            model.zero_grad()   # clean the grad

            i_sample, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)
            
            if args.use_gpu:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()

            pred, hiddens = model(text, audio, vision)
            batch_loss = criterion(pred, eval_attr)
            batch_loss.backward()   # backprop grad
            optimizer.step()    # update the para

    def evaluate(test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        
        total_loss = 0
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                i_sample, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)

                batch_size = text.size(0)

                if args.use_gpu:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                

                pred, _ = model(text, audio, vision)
                batch_loss = criterion(pred, eval_attr).item() * batch_size
                total_loss += batch_loss

                results.append(pred)
                truths.append(eval_attr)
            
        total_len = args.test_len if test else args.valid_len
        avg_loss = total_loss / total_len
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths
        
    ### train loop 
    print('Training...')
    if args.training:
        best_loss = 1e8
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.1, verbose=True)
        for i in range(args.epoch):
            start_time = time.time()
            train()
            valid_loss, valid_results, valid_truths = evaluate(False)
            test_loss, test_results, test_truths = evaluate(True)
            end_time = time.time()
            duration = end_time - start_time
            scheduler.step(valid_loss)
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(i+1, duration, valid_loss, test_loss))
            print("-"*50)

            ### save best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                print(f"Saved model at {args.savepath}/{args.model}.pt!")
                save_path = os.path.join(args.savepath, args.model) + '.pt'
                torch.save(model, save_path)

    model = torch.load(os.path.join(args.savepath, args.model) + '.pt')
    _, results, truths = evaluate(True)
    ### evaluate
    print('Evaluating...')
    if args.dataset == "mosei":
        eval_mosei_senti(results, truths, True)
    elif args.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif args.dataset == 'iemocap':
        eval_iemocap(results, truths)




