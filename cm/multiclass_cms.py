from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, accuracy_score, roc_curve)
import matplotlib as plt

def multiclass_cm_with_percents(proba_test, y_test_c, ypt, class_names, cmap='virids'):
  ''' 
  cmap="Blues" is a good alternative
  '''
  n_classes = proba_test.shape[1]
  if class_names is None:
      class_names = [f"class_{i}" for i in range(n_classes)]
  #print(classification_report(y_test_c, ypt, target_names=class_names))
  
  cm = confusion_matrix(y_test_c, ypt, labels=list(range(n_classes)))
  cm_pct = cm / cm.sum(axis=1, keepdims=True)

  disp = ConfusionMatrixDisplay(confusion_matrix=cm_pct, display_labels=class_names)
  disp.plot(xticks_rotation=45, include_values=False, cmap=cmap)
        
  high_color = "black"
  low_color = "white"
  if "Blues" in cmap:
      high_color = "white"
      low_color = "black"
  elif "viridis_r" in cmap:
      high_color = "purple" 
      low_color = "yellow"
       
  for i in range(n_classes):
      for j in range(n_classes):
          pct = cm_pct[i, j] * 100
          color = low_color if cm_pct[i, j] < 0.5 else high_color
          disp.ax_.text(j, i, f"{cm[i, j]}\n{pct:.1f}%", ha="center", va="center", 
                        color=color, fontsize=11)
  return cm
