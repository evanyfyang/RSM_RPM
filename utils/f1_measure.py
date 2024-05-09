class F1_Measure:
    def __init__(self):
        self.pred_list = []
        self.true_list = []

    def pred_inc(self, idx, preds):
        for pred in preds:
            self.pred_list.append((idx, pred))
            
    def true_inc(self, idx, trues):
        for true in trues:
            self.true_list.append((idx, true))
            
    def report(self):
        self.f1, self.p, self.r = self.cal_f1(self.pred_list, self.true_list)
        return self.f1
    
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise NotImplementedError

    def cal_f1(self, pred_list, true_list):
        n_tp = 0
        for pred in pred_list:
            if pred in true_list:
                n_tp += 1    
        _p = n_tp / len(pred_list) if pred_list else 1
    
        n_tp = 0
        for true in true_list:
            if true in pred_list:
                n_tp += 1 
        _r = n_tp / len(true_list) if true_list else 1

        f1 = 2 * _p * _r / (_p + _r) if _p + _r else 0

        return f1, _p, _r