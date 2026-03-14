import numpy as np

def get_tp_fp_fn_tn(cm):
    cm = np.asarray(cm, dtype=float)

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    return tp, fp, fn, tn

def get_precision_and_recall(tp, fp, fn):
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    return precision, recall

def get_F1_per_class(precision, recall):
    f1_per_class = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0)
    return f1_per_class

def get_accuracy_per_class(tp, fp, fn, tn):
    acc_per_class = np.divide(
        tp + tn,
        tp + tn + fp + fn,
        out=np.zeros_like(tp),
        where=(tp + tn + fp + fn) != 0
    )
    return acc_per_class
    
def macro_F1_from_cm(cm):
    tp, fp, fn, tn = get_tp_fp_fn_tn(cm)
    p, r = get_precision_and_recall(tp, fp, fn) 
    f1_per_class = get_F1_per_class(p, r)
    macro_f1 = f1_per_class.mean()
    return macro_f1, f1_per_class

def macro_F1_accuracy_from_cm(cm):
    tp, fp, fn, tn = get_tp_fp_fn_tn(cm) 
    p, r = get_precision_and_recall(tp, fp, fn) 
    f1_per_class = get_F1_per_class(p, r)
    acc_per_class = get_accuracy_per_class(tp, fp, fn, tn)
    macro_f1 = f1_per_class.mean()
    macro_acc = acc_per_class.mean()
    return macro_f1, macro_acc, f1_per_class, acc_per_class
