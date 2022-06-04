# IMPORTS
import skimage, sklearn, matplot, scipy, pickle, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import time
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import roc_curve, auc, average_precision_score, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay, plot_precision_recall_curve, precision_recall_curve, average_precision_score
from numpy import reshape
from mlxtend.evaluate import mcnemar_table, mcnemar

# LOADS THE CONTENT OF THE EMBEDDING FILE, CLASSES FILE AND PATCH_INFORMATION FILE
def load_files(folder, wanted_content):
    content = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            #load embeddings and classes
            if name.endswith('.npy') and name == wanted_content:
                file = np.load(os.path.join(root, name))
                for array in file:
                    content.append(array)
            #load information about patches
            elif name.endswith('.pkl') and name == wanted_content:
                with open(os.path.join(root, name), 'rb') as f:
                    content = pickle.load(f)
    return content

# LOAD ALL DATA FROM A FOLDER
def load_data(folder):
    embeddings = load_files(folder, 'embeddings.npy')
    classes = get_class_array(load_files(folder, 'classes.npy'))
    patch_information = load_files(folder, 'paths.pkl')
    return embeddings, classes, patch_information

# PRINTS ALL INFORMATION ABOUT AN EMBEDDING
def print_data(i, embeddings, classes, patch_information):    
    print('---------------- EMBEDDING -----------------------')
    print(embeddings[i])
    print('---------------- CLASS -----------------------')
    print(classes[i])
    print('---------------- PATCH INFORMATION -----------------------')
    print(patch_information[i])

# FIXES THE ARRAY STRUCTURE OF THE CLASSES ARRAY
def get_class_array(array):
    lst = []
    for x in array:
        for y in x:
            lst.append(y)
    return lst

# SORTS THE TRAIN EMBEDDINGS FROM THE TEST EMBEDDINGS
def sort_train_test(embeddings_array, class_array, info):
    train_embeddings = []
    test_embeddings = []
    train_classes = []
    test_classes = []
    patch_info_train = []
    patch_info_test = []
    i = 0
    for embedding in embeddings_array:
        if "train" in info[i]:
            train_embeddings.append(embedding)
            train_classes.append(class_array[i])
            patch_info_train.append(info[i])
        elif "test" in info[i]:
            test_embeddings.append(embedding)
            test_classes.append(class_array[i])
            patch_info_test.append(info[i])
        else:
            print("patch info error")
        i += 1
    return train_embeddings, test_embeddings, train_classes, test_classes, patch_info_train, patch_info_test

# EVALUATE A NETWORK
def evaluate_network(network, network_name, embeddings_test, classes_test):
    print('ACCURACY:')
    print(network.score(embeddings_test, classes_test))
    print('AUC & ROC CURVE:')
    ROC(network, embeddings_test, classes_test)
    print('PRECISION RECALL CURVE:')
    PRC(network, embeddings_test, classes_test)
    print('CONFUSION MATRIX:')
    confusion_matrix_plot(network_name, network, embeddings_test, classes_test)
    print('---------------------------------------')

# PLOT CONFUSION MATRIX FOR A NETWORK
def confusion_matrix_plot(network_name, network, embeddings_test, classes_test):
    predictions = network.predict(embeddings_test)
    cm = confusion_matrix(classes_test, predictions, labels=network.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=network.classes_)
    disp.plot(cmap=plt.cm.Greens)
    plt.savefig('results/confusion_matrix_'+network_name+'.jpg')
    plt.show()

# PLOT PRECISION RECALL CURVE AND AP FOR A NETWORK
def PRC(network, embeddings_test, classes_test):
    y_score = network.predict_log_proba(embeddings_test)[::,1]
    display = PrecisionRecallDisplay.from_predictions(classes_test, y_score, name="LinearSVC")
    display.ax_.set_aspect('equal', adjustable='box')
    display.ax_.set_title("2-class Precision-Recall curve")
    plt.show()

# PLOT ROC AND AUC FOR A NETWORK
def ROC(network, embeddings_test, classes_test):
    y_pred_proba = network.predict_proba(embeddings_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(classes_test,  y_pred_proba)
    auc = metrics.roc_auc_score(classes_test, y_pred_proba)
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.axis('square')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

# PLOT THE ACCURACY, ROC, AUC, AP AND PRECISION RECALL FOR ALL MLP AND LR NETWORKS
def plot_result(LR_inpainting, LR_simclr_light, LR_simclr_aggressive, MLP_inpainting, MLP_simclr_light, MLP_simclr_aggressive, GB_inpainting, embeddings_inpainting_test, classes_inpainting_test, embeddings_simclr_light_test, classes_simclr_light_test, embeddings_simclr_aggressive_test, classes_simclr_aggressive_test):
    print('ACCURACIES FOR EACH NETWORK:')
    print('LR with DIME-Inpainting: '+str(LR_inpainting.score(embeddings_inpainting_test, classes_inpainting_test)))
    print('LR with DIME-SimCLR Light: '+str(LR_simclr_light.score(embeddings_simclr_light_test, classes_simclr_light_test)))
    print('LR with DIME-SimCLR Aggressive: '+str(LR_simclr_aggressive.score(embeddings_simclr_aggressive_test, classes_simclr_aggressive_test)))
    print('MLP with DIME-Inpainting: '+str(MLP_inpainting.score(embeddings_inpainting_test, classes_inpainting_test)))
    print('MLP with DIME-SimCLR Light: '+str(MLP_simclr_light.score(embeddings_simclr_light_test, classes_simclr_light_test)))
    print('MLP with DIME-SimCLR Aggressive: '+str(MLP_simclr_aggressive.score(embeddings_simclr_aggressive_test, classes_simclr_aggressive_test)))
    print('AUC & ROC CURVE:') 
    network_lst = [(LR_inpainting, 'LR with DIME-Inpainting'), (LR_simclr_light, 'LR with DIME-SimCLR Light'), (LR_simclr_aggressive, 'LR with DIME-SimCLR Aggressive'), (MLP_inpainting, 'MLP with DIME-Inpainting'), (MLP_simclr_light, 'MLP with DIME-SimCLR Light'), (MLP_simclr_aggressive, 'MLP with DIME-SimCLR Aggressive')]
    ROC_multiple_networks(network_lst, embeddings_inpainting_test, classes_inpainting_test, embeddings_simclr_light_test, classes_simclr_light_test, embeddings_simclr_aggressive_test, classes_simclr_aggressive_test)
    print('PRECISION RECALL CURVE:')
    PRC_multiple_networks(network_lst, embeddings_inpainting_test, classes_inpainting_test, embeddings_simclr_light_test, classes_simclr_light_test, embeddings_simclr_aggressive_test, classes_simclr_aggressive_test)

# PLOT ROC AND AUC FOR MULTIPLE NETWORKS
def ROC_multiple_networks(network_lst, embeddings_inpainting_test, classes_inpainting_test, embeddings_simclr_light_test, classes_simclr_light_test, embeddings_simclr_aggressive_test, classes_simclr_aggressive_test):
    for network in network_lst:
        if 'Inpainting' in network[1]:
            y_pred_proba = network[0].predict_proba(embeddings_inpainting_test)[::,1]
            fpr, tpr, _ = metrics.roc_curve(classes_inpainting_test,  y_pred_proba)
            auc = metrics.roc_auc_score(classes_inpainting_test, y_pred_proba)
            plt.plot(fpr,tpr,label=str(network[1])+' (AUC: '+str(round(auc,2))+')')
        elif 'SimCLR Light' in network[1]:
            y_pred_proba = network[0].predict_proba(embeddings_simclr_light_test)[::,1]
            fpr, tpr, _ = metrics.roc_curve(classes_simclr_light_test,  y_pred_proba)
            auc = metrics.roc_auc_score(classes_simclr_light_test, y_pred_proba)
            plt.plot(fpr,tpr,label=str(network[1])+' (AUC: '+str(round(auc,2))+')')
        else:
            y_pred_proba = network[0].predict_proba(embeddings_simclr_aggressive_test)[::,1]
            fpr, tpr, _ = metrics.roc_curve(classes_simclr_aggressive_test,  y_pred_proba)
            auc = metrics.roc_auc_score(classes_simclr_aggressive_test, y_pred_proba)
            plt.plot(fpr,tpr,label=str(network[1])+' (AUC: '+str(round(auc,2))+')')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.savefig('results/roc_all_models.png', dpi=300, bbox_inches='tight')
    plt.show()

# PLOT PRECISION RECALL CURVE AND AP FOR MULTIPLE NETWORKS
def PRC_multiple_networks(network_lst, embeddings_inpainting_test, classes_inpainting_test, embeddings_simclr_light_test, classes_simclr_light_test, embeddings_simclr_aggressive_test, classes_simclr_aggressive_test):  
    for network in network_lst:
        if 'Inpainting' in network[1]:
            y_pred_proba = network[0].predict_proba(embeddings_inpainting_test)[::,1]
            precision, recall, thresholds = precision_recall_curve(classes_inpainting_test, y_pred_proba)
            ap = average_precision_score(classes_inpainting_test, y_pred_proba)
            plt.plot(recall, precision, label=str(network[1])+' (AP: '+str(round(ap,2))+')')
        elif 'SimCLR Light' in network[1]:
            y_pred_proba = network[0].predict_proba(embeddings_simclr_light_test)[::,1]
            precision, recall, thresholds = precision_recall_curve(classes_simclr_light_test, y_pred_proba)
            ap = average_precision_score(classes_inpainting_test, y_pred_proba)
            plt.plot(recall, precision, label=str(network[1])+' (AP: '+str(round(ap,2))+')')
        else:
            y_pred_proba = network[0].predict_proba(embeddings_simclr_aggressive_test)[::,1]
            precision, recall, thresholds = precision_recall_curve(classes_simclr_aggressive_test, y_pred_proba)
            ap = average_precision_score(classes_inpainting_test, y_pred_proba)
            plt.plot(recall, precision, label=str(network[1])+' (AP: '+str(round(ap,2))+')')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc='lower left');
    plt.savefig('results/precision_recall_all_models.png', dpi=300, bbox_inches='tight')
    plt.show()

# GET A REDUCED DATASET THE SIZE OF PERCENTAGE * ORIGINAL SIZE
def percentage_dataset(X, y, percentage):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percentage, random_state=42)
    return X_test, y_test

# MCNEMARS TEST
def mcnemar_test(data):
    chi2, p = mcnemar(ary=data, exact=False, corrected=False)
    print('chi-squared:', chi2)
    print('p-value:', p)

# PRODUCE CONTINGENCY TABLE
def mcnemar_data_table(model1, model2, embeddings_model1_test, embeddings_model2_test, classes_test, name):
    y_model1 = np.array(model1.predict(embeddings_model1_test))
    y_model2 = np.array(model2.predict(embeddings_model2_test))
    both_correct = 0
    none_correct = 0
    model1_correct = 0
    model2_correct = 0
    i = 0
    for x in classes_test:
        if x == y_model1[i] and x == y_model2[i]:
            both_correct += 1
        elif x == y_model2[i]:
            model2_correct += 1
        elif x == y_model1[i]:
            model1_correct += 1
        else: 
            none_correct += 1
        i += 1
    data = np.array([[both_correct, model2_correct],[model1_correct, none_correct]])
    plot_contingency_table(name, data)
    return data

# PLOT AUC FOR EACH REDUCED TRAINING SET MODEL
def percentage_plot_auc(network_lst, embeddings_inpainting_test, classes_inpainting_test, embeddings_simclr_light_test, classes_simclr_light_test, embeddings_simclr_aggressive_test, classes_simclr_aggressive_test):  
    percentage_lst = [0.1,0.3,0.5,0.7,1]
    plot_percentage_lst = [10,30,50,70,100]
    for emb_type in network_lst:
        if emb_type == 'LR with DIME-Inpainting':
            auc_list = []
            for percentage in percentage_lst:
                network = load_model('models/LR_inpainting_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_inpainting_test)[::,1]
                auc = metrics.roc_auc_score(classes_inpainting_test, y_pred_proba)
                auc_list.append(auc)
            plt.plot(plot_percentage_lst,auc_list,label=str(emb_type),ls='--',marker = '.', markersize = 10)
        elif emb_type == 'MLP with DIME-Inpainting':
            auc_list = []
            for percentage in percentage_lst:
                network = load_model('models/MLP_inpainting_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_inpainting_test)[::,1]
                auc = metrics.roc_auc_score(classes_inpainting_test, y_pred_proba)
                auc_list.append(auc)
            plt.plot(plot_percentage_lst,auc_list,label=str(emb_type),ls='--',marker = '.', markersize = 10)  
        elif emb_type == 'GB with DIME-Inpainting':
            auc_list = []
            for percentage in percentage_lst:
                network = load_model('models/GB_inpainting_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_inpainting_test)[::,1]
                auc = metrics.roc_auc_score(classes_inpainting_test, y_pred_proba)
                auc_list.append(auc)
            plt.plot(plot_percentage_lst,auc_list,label=str(emb_type),ls='--',marker = '.', markersize = 10)
        elif emb_type == 'LR with DIME-SimCLR Light':
            auc_list = []
            for percentage in percentage_lst:
                network = load_model('models/LR_simclr_light_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_simclr_light_test)[::,1]
                auc = metrics.roc_auc_score(classes_inpainting_test, y_pred_proba)
                auc_list.append(auc)
            plt.plot(plot_percentage_lst,auc_list,label=str(emb_type),ls='--',marker = '.', markersize = 10)
        elif emb_type == 'MLP with DIME-SimCLR Light':
            auc_list = []
            for percentage in percentage_lst:
                network = load_model('models/MLP_simclr_light_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_simclr_light_test)[::,1]
                auc = metrics.roc_auc_score(classes_inpainting_test, y_pred_proba)
                auc_list.append(auc)
            plt.plot(plot_percentage_lst,auc_list,label=str(emb_type),ls='--',marker = '.', markersize = 10)
        elif emb_type == 'LR with DIME-SimCLR Aggressive':
            auc_list = []
            for percentage in percentage_lst:
                network = load_model('models/LR_simclr_aggressive_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_simclr_aggressive_test)[::,1]
                auc = metrics.roc_auc_score(classes_inpainting_test, y_pred_proba)
                auc_list.append(auc)
            plt.plot(plot_percentage_lst,auc_list,label=str(emb_type),ls='--',marker = '.', markersize = 10)
        else:
            auc_list = []
            for percentage in percentage_lst:
                network = load_model('models/MLP_simclr_aggressive_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_simclr_aggressive_test)[::,1]
                auc = metrics.roc_auc_score(classes_inpainting_test, y_pred_proba)
                auc_list.append(auc)
            plt.plot(plot_percentage_lst,auc_list,label=str(emb_type),ls='--',marker = '.', markersize = 10)
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,0.88,0.96))
    plt.ylabel('AUC')
    plt.xlabel('Percentage')
    plt.legend(loc='lower right');
    plt.savefig('results/auc_percentage.png', dpi=300, bbox_inches='tight')
    plt.show()

# PLOT AP FOR EACH REDUCED TRAINING SET MODEL
def percentage_plot_ap(network_lst, embeddings_inpainting_test, classes_inpainting_test, embeddings_simclr_light_test, classes_simclr_light_test, embeddings_simclr_aggressive_test, classes_simclr_aggressive_test):  
    percentage_lst = [0.1,0.3,0.5,0.7,1]
    plot_percentage_lst = [10,30,50,70,100]
    for emb_type in network_lst:
        if emb_type == 'LR with DIME-Inpainting':
            ap_list = []
            for percentage in percentage_lst:
                network = load_model('models/LR_inpainting_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_inpainting_test)[::,1]
                ap = average_precision_score(classes_inpainting_test, y_pred_proba)
                ap_list.append(ap)
            plt.plot(plot_percentage_lst,ap_list,label=str(emb_type),ls='--',marker = '.', markersize = 10)
        elif emb_type == 'MLP with DIME-Inpainting':
            ap_list = []
            for percentage in percentage_lst:
                network = load_model('models/MLP_inpainting_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_inpainting_test)[::,1]
                ap = average_precision_score(classes_inpainting_test, y_pred_proba)
                ap_list.append(ap)
            plt.plot(plot_percentage_lst,ap_list,label=str(emb_type),ls='--',marker = '.', markersize = 10) 
        elif emb_type == 'GB with DIME-Inpainting':
            ap_list = []
            for percentage in percentage_lst:
                network = load_model('models/GB_inpainting_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_inpainting_test)[::,1]
                ap = average_precision_score(classes_inpainting_test, y_pred_proba)
                ap_list.append(ap)
            plt.plot(plot_percentage_lst,ap_list,label=str(emb_type),ls='--',marker = '.', markersize = 10)
        elif emb_type == 'LR with DIME-SimCLR Light':
            ap_list = []
            for percentage in percentage_lst:
                network = load_model('models/LR_simclr_light_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_simclr_light_test)[::,1]
                ap = average_precision_score(classes_inpainting_test, y_pred_proba)
                ap_list.append(ap)
            plt.plot(plot_percentage_lst,ap_list,label=str(emb_type),ls='--',marker = '.', markersize = 10) 
        elif emb_type == 'MLP with DIME-SimCLR Light':
            ap_list = []
            for percentage in percentage_lst:
                network = load_model('models/MLP_simclr_light_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_simclr_light_test)[::,1]
                ap = average_precision_score(classes_inpainting_test, y_pred_proba)
                ap_list.append(ap)
            plt.plot(plot_percentage_lst,ap_list,label=str(emb_type),ls='--',marker = '.', markersize = 10)
        elif emb_type == 'LR with DIME-SimCLR Aggressive':
            ap_list = []
            for percentage in percentage_lst:
                network = load_model('models/LR_simclr_aggressive_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_simclr_aggressive_test)[::,1]
                ap = average_precision_score(classes_inpainting_test, y_pred_proba)
                ap_list.append(ap)
            plt.plot(plot_percentage_lst,ap_list,label=str(emb_type),ls='--',marker = '.', markersize = 10)
        else:
            ap_list = []
            for percentage in percentage_lst:
                network = load_model('models/MLP_simclr_aggressive_'+str(percentage)+'.sav')
                y_pred_proba = network.predict_proba(embeddings_simclr_aggressive_test)[::,1]
                ap = average_precision_score(classes_inpainting_test, y_pred_proba)
                ap_list.append(ap)
            plt.plot(plot_percentage_lst,ap_list,label=str(emb_type),ls='--',marker = '.', markersize = 10)
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,0.77,1))
    plt.ylabel('AP')
    plt.xlabel('Percentage')
    plt.legend(loc='upper left');
    plt.savefig('results/ap_percentage.png', dpi=300, bbox_inches='tight')
    plt.show()    

# PLOT CONTINGENCY TABLE
def plot_contingency_table(name, table):
    fig = plt.figure()
    plt.clf()
    res = sns.heatmap(table, annot=True, fmt="d", cmap=plt.cm.Greens)
    plt.yticks([0.5,1.5], ['Model 2 Correct', 'Model 2 Wrong'],va='center')
    plt.xticks([0.5,1.5], ['Model 1 Correct', 'Model 1 Wrong'],va='center')
    plt.savefig('results/contingency_table_'+name+'.jpg')
    plt.show()

# CROSS VALIDATION OF MODELS
def cross_validation(network, X, y, fold=5):
    scores = cross_val_score(network, X, y, cv=fold, n_jobs=-1)
    i = 0
    for score in scores:
        print(f"Accuracy for the fold no. {i} on the test set: {score}")
        i += 1
    print("TOTAL SCORE: ")
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    
# NUMBER OF TUMOUR EMBEDDINGS
def number_tumour_embeddings(classes):
    tumours = 0
    for x in classes:
        if x == 1:
            tumours += 1
    return tumours

# GIVES A LIST OF MPP LEVELS IN THE DATASET
def number_zoom_levles(patch_info):
    mpp_lst = []
    for x in patch_info:
        x1 = x.split("/")
        x2 = x1[3].split(",")
        if x2[0] in mpp_lst:
            continue
        else: 
            mpp_lst.append(x2[0])
    return mpp_lst

# RUN A GRADIENT BOOSTING NETWORK
def runGB(X, y, X_t, y_t, learning_rate, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, percentage=0.1):
    if percentage != 1:
        X_train, y_train = percentage_dataset(X, y, percentage)
    else:
        X_train = X
        y_train = y
    start = time.time()
    from sklearn.ensemble import GradientBoostingClassifier
    network = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
    network.fit(X_train, y_train)
    filename = 'models/GB_inpainting_'+str(percentage)+'.sav'
    save_model(filename, network)
    end = time.time()
    tot = end - start
    num_emb = len(X_train)
    print("---------------------------------------")
    print("PERCENTAGE: "+str(percentage*100)+"% OF DATASET, NUMBER EMBEDDINGS: "+str(num_emb))
    print("HYPERPARAMETERS: learning_rate = "+str(learning_rate)+", n_estimators = "+str(n_estimators)+", max_depth = "+str(max_depth)+", min_samples_split = "+str(min_samples_split)+",  min_samples_leaf = "+str( min_samples_leaf)+", max_features = "+str(max_features))
    print("RUNTIME IN SEC: "+str(tot))
    evaluate_network(network, X_t, y_t)

# SAVE A MODEL
def save_model(filename, network):
    pickle.dump(network, open(filename, 'wb'))
    
# LOAD A SAVED MODEL
def load_model(filename):
    return pickle.load(open(filename, 'rb'))

# RUN MULTIPLE GRADIENT BOOSTING NETWORKS WITH DIFFERENT TRAIN SET PERCENTAGES
def runGBPercentage(X, y, X_t, y_t, percentageLst, learning_rate=0.1, n_estimators=100, max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=None):
    for percentage in percentageLst:
        runGB(X, y, X_t, y_t, learning_rate, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, percentage)

# RUN A LOGISTIC REGRESSION NETWORK
def runLR(X, y, X_t, y_t, sol, pen, c, percentage=0.1, ite=200, jobs=8):
    if percentage != 1:
        X_train, y_train = percentage_dataset(X, y, percentage)
    else:
        X_train = X
        y_train = y
    start = time.time()
    from sklearn.linear_model import LogisticRegression
    network = LogisticRegression(max_iter=ite, n_jobs=jobs, solver=sol, penalty=pen, C=c)
    network.fit(X_train, y_train)
    filename = 'models/LR_inpainting_'+str(percentage)+'.sav'
    save_model(filename, network)
    end = time.time()
    tot = end - start
    num_emb = len(X_train)
    print("---------------------------------------")
    print("PERCENTAGE: "+str(percentage*100)+"% OF DATASET, NUMBER EMBEDDINGS: "+str(num_emb))
    print("HYPERPARAMETERS: C = "+str(c)+", SOLVER = "+str(sol)+", PENALTY = "+str(pen))
    print("RUNTIME IN SEC: "+str(tot))
    evaluate_network(network, X_t, y_t)  

# RUN MULTIPLE LOGISTIC REGRESSION NETWORKS WITH DIFFERENT TRAIN SET PERCENTAGES
def runLRPercentage(X, y, X_t, y_t, percentageLst, sol='lbfgs', pen='l2', c=1.0):
    for percentage in percentageLst:
        runLR(X, y, X_t, y_t, sol, pen, c, percentage)
        
# FIND BEST HYPERPARAMETERS FOR A LR NETWORK
def runLRHypPar(X, y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=200)
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'saga', 'sag']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='roc_auc',error_score=0)
    grid_result = grid_search.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_params_['C'], grid_result.best_params_['penalty'], grid_result.best_params_['solver']

# RUN A MULTILAYER PERCEPTRON NETWORK
def run_MLP(X, y, X_t, y_t, hid_lay, act, solv, alph, lr, percentage=0.1, ite=200):
    if percentage != 1:
        X_train, y_train = percentage_dataset(X, y, percentage)
    else:
        X_train = X
        y_train = y
    start = time.time()
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(max_iter=ite, hidden_layer_sizes=hid_lay, activation=act, solver=solv, alpha=alph, learning_rate=lr)
    mlp.fit(X_train, y_train)
    filename = 'models/MLP_simclr_aggressive_'+str(percentage)+'.sav'
    save_model(filename, mlp)
    end = time.time()
    tot = end - start
    num_emb = len(X_train)
    print("---------------------------------------")
    print("PERCENTAGE: "+str(percentage*100)+"% OF DATASET, NUMBER EMBEDDINGS: "+str(num_emb))
    print("HYPERPARAMETERS: HIDDEN LAYER SIZES = "+str(hid_lay)+", ACTIVATION = "+str(act)+", SOLVER = "+str(solv)+", ALPHA = "+str(alph)+", LEARNING RATE = "+str(lr))
    print("RUNTIME IN SEC: "+str(tot))
    evaluate_network(mlp, X_t, y_t)

# RUN MULTIPLE MULTILAYER PERCEPTRON NETWORKS WITH DIFFERENT TRAIN SET PERCENTAGES
def run_Percentage_MLP(X, y, X_t, y_t, percentageLst, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, learning_rate='constant'):
    for percentage in percentageLst:
        run_MLP(X, y, X_t, y_t, hidden_layer_sizes, activation, solver, alpha, learning_rate, percentage)
    
# FIND BEST HYPERPARAMETERS FOR A MLP NETWORK
def run_HypPar_MLP(X, y):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(max_iter=200)
    hidden_layer_sizes = [(50,50,50), (50,100,50), (100,)]
    activation = ['tanh', 'relu']
    solver = ['sgd', 'adam']
    # choose hyp.parm and explain in report
    alpha = [0.0001, 0.05]
    learning_rate = ['constant','adaptive']
    # define grid search
    grid = dict(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='roc_auc',error_score=0)
    grid_result = grid_search.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_params_['hidden_layer_sizes'], grid_result.best_params_['activation'], grid_result.best_params_['solver'],grid_result.best_params_['alpha'], grid_result.best_params_['learning_rate']