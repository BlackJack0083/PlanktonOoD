### `openood/evaluators/metrics.py`

```python
def compute_all_metrics(conf, label, pred):
    np.set_printoptions(precision=3)
    recall_95 = 0.95, recall_99 = 0.99
    auroc, aupr_in, aupr_out, fpr95_id, fpr99_id, fpr95_ood, fpr99_ood = auc_and_fpr_recall(conf, label, recall_95, recall_99)

    accuracy = acc(pred, label)

    #caculate your new metrics and put them here.
    results = [fpr95_id, fpr99_id, fpr95_ood, fpr99_ood, auroc, aupr_in, aupr_out, accuracy] 

    return results
```

### `openood/evaluation_api/evaluator.py`

#### `eval_ood`

```python
            self.metrics[task] = pd.DataFrame(
                np.concatenate([near_metrics, far_metrics], axis=0),
                index=list(self.dataloader_dict['ood']['near'].keys()) +
                ['nearood'] + list(self.dataloader_dict['ood']['far'].keys()) +
                ['farood'],
                columns=['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC'], # modify here.
            )
```

#### `_print_metrics`

```python
    def _print_metrics(self, metrics):
        [fpr95_id, fpr99_id, fpr95_ood, fpr99_ood, auroc, aupr_in, aupr_out, accuracy] = metrics

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print(u'\u2500' * 70, flush=True)
        print('', flush=True)
```