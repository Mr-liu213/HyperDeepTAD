import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"可用 GPU: {[gpu.name for gpu in gpus]}")
else:
    print("无可用 GPU，将使用 CPU")
if gpus:
    try:      
        tf.config.set_visible_devices(gpus[4], 'GPU')
       
        visible_devices = tf.config.get_visible_devices('GPU')
        print(f"已指定使用 GPU: {[d.name for d in visible_devices]}")
    except RuntimeError as e:
      
        print(e)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已启用 GPU 内存动态增长")
    except RuntimeError as e:
        print(e)

#F1Score
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum(y_pred) - tp
        fn = tf.reduce_sum(y_true) - tp
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-6)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-6)
        return 2 * (precision * recall) / (precision + recall + 1e-6)

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


#AdaptiveFocalLoss
class AdaptiveFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma, alpha, delta,
                 reduction=tf.keras.losses.Reduction.AUTO, name='AdaptiveFocalLoss'):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        p_t = tf.clip_by_value(p_t, self.delta, 1.0 - self.delta)
        alpha_factor = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_loss = -alpha_factor * tf.pow(1 - p_t, self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "delta": self.delta
        })
        return config



# dynamic convolutional layer
class DynamicConv1D(layers.Layer):
    """Dynamic convolutional layer that fuses multiple expert convolution kernels"""
    def __init__(self, filters, kernel_size, strides=1, padding='same', num_experts=4, **kwargs):
        super(DynamicConv1D, self).__init__(** kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.num_experts = num_experts
        
        self.experts = [
            layers.Conv1D(filters, kernel_size, strides=strides, padding=padding,
                         kernel_regularizer=l2(1e-4), use_bias=False)
            for _ in range(num_experts)
        ]
        
        self.gate = tf.keras.Sequential([
            layers.GlobalAveragePooling1D(),
            layers.Dense(num_experts, activation='softmax', kernel_regularizer=l2(1e-4))
        ])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        gates = self.gate(inputs)  # [batch, num_experts]
        gates = tf.reshape(gates, [batch_size, 1, 1, self.num_experts])
        
        expert_outputs = [tf.expand_dims(expert(inputs), axis=-1) for expert in self.experts]
        expert_outputs = tf.concat(expert_outputs, axis=-1)
        return tf.reduce_sum(expert_outputs * gates, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters, 'kernel_size': self.kernel_size,
            'strides': self.strides, 'padding': self.padding, 'num_experts': self.num_experts
        })
        return config

class Module(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same',
                 num_experts=4, **kwargs):  
        super(Module, self).__init__(** kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.num_experts = num_experts
        

        self.dynamic_conv = tf.keras.Sequential([
            DynamicConv1D(filters, kernel_size, strides, padding, num_experts),
            layers.BatchNormalization(), layers.ReLU()
        ])
        
     
        self.fusion = layers.Conv1D(filters, 1, kernel_regularizer=l2(1e-4))
        self.bn_fusion = layers.BatchNormalization()
        self.residual = layers.Conv1D(filters, 1, strides=strides, kernel_regularizer=l2(1e-4)) \
                        if strides != 1 else None
        
    def call(self, inputs):
        conv_out = self.dynamic_conv(inputs)

        out = self.bn_fusion(self.fusion(conv_out))  
        
        if self.residual:
            out = layers.add([out, self.residual(inputs)])
        return tf.nn.relu(out)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters, 'kernel_size': self.kernel_size, 'strides': self.strides,
            'padding': self.padding, 'num_experts': self.num_experts
        })
        return config

# attention mechanism
def hierarchical_attention(inputs):
    time_att = tf.nn.softmax(layers.Dense(1, activation='tanh')(inputs), axis=1)
    time_context = tf.reduce_sum(inputs * time_att, axis=1)
    
    feat_att = tf.nn.softmax(layers.Dense(inputs.shape[-1], activation='tanh')(inputs), axis=2)
    feat_context = tf.reduce_sum(inputs * feat_att, axis=1)
    
    return (time_context + feat_context) / 2

# model construction function
def build_optimized_model(input_shape,lr):
    inputs = layers.Input(shape=input_shape)

    x = Module(256, kernel_size=5)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    x = Module(128, kernel_size=3)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    gru_out = layers.Bidirectional(layers.GRU(
        64, return_sequences=True, dropout=0.4, recurrent_dropout=0.4,
        kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4)
    ))(x)
    res_out = tf.keras.layers.add([x, gru_out])
    # res_out = LayerNormalization()(res_out)
    res_out = layers.LayerNormalization()(res_out)

    attention_out = hierarchical_attention(res_out)

    x = layers.Dense(128, kernel_regularizer=l2(1e-4))(attention_out)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(64, kernel_regularizer=l2(1e-4))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=AdaptiveFocalLoss(gamma=2.0, alpha=0.5, delta=0.02),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.Precision(name='precision'),
                 F1Score()]
    )
    return model

#Calculate the degree of nodes
def compute_node_degrees(H):
    return np.sum(H, axis=1)

# Calculate the degree of hyperedges
def compute_hyperedge_degrees(H):
    return np.sum(H, axis=0)

# Calculate the first-order transition probability
def compute_first_order_transition_probabilities(H, node_degrees, hyperedge_degrees):
    num_nodes, num_hyperedges = H.shape
    P1 = np.zeros((num_nodes, num_nodes))
    for v in range(num_nodes):
        for u in range(num_nodes):
            if u == v:
                continue

            pi_uv = 0
            for e in range(num_hyperedges):
                if H[u, e] == 0 or H[v, e] == 0:  
                    continue

                h_ve = H[v, e]  
                h_ue = H[u, e]  
                d_v = node_degrees[v]  
                delta_e = hyperedge_degrees[e]  
                pi_uv += (h_ve * h_ue) / (d_v * delta_e)

            P1[v, u] = pi_uv

    return P1



#Load data
def process_files_to_arrays(filenames):
    X_all_chr = []
    middle_row_indices_all_chr = []
    y_all_chr = []
    sample_counts = []  

    for filename in filenames:
        count = 0
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    data = eval(line, {"array": np.array})
                    X_all_chr.append(data[0])
                    middle_row_indices_all_chr.append(data[1])
                    y_all_chr.append(data[2])
                    count += 1
        sample_counts.append(count)

    return (np.array(X_all_chr),
            np.array(middle_row_indices_all_chr),
            np.array(y_all_chr),
            sample_counts)

#positive and negative samples
def select(X_train, middle_row_indices_train, y_train, safe_radius):
    positive_mask = (y_train == 1)
    negative_mask = ~positive_mask

    positive_indices = np.where(positive_mask)[0]
    negative_indices = np.where(negative_mask)[0]

    if len(positive_indices) == 0:
        return X_train, middle_row_indices_train, y_train

    positive_positions = middle_row_indices_train[positive_indices]
    negative_positions = middle_row_indices_train[negative_indices]

    distances = np.abs(negative_positions[:, np.newaxis] - positive_positions)
    min_distances = np.min(distances, axis=1)

    safe_mask = min_distances > safe_radius
    safe_negative_indices = negative_indices[safe_mask]

    n_pos = len(positive_indices)
    n_neg_desired =  n_pos

    if len(safe_negative_indices) >= n_neg_desired:
        selected_neg = np.random.choice(safe_negative_indices, n_neg_desired, replace=False)
    else:
        selected_neg = np.random.choice(safe_negative_indices, n_neg_desired, replace=True)

    selected_indices = np.concatenate([positive_indices, selected_neg])
    np.random.shuffle(selected_indices)

    return (
        X_train[selected_indices],
        middle_row_indices_train[selected_indices],
        y_train[selected_indices]
    )



def generate_P(data):
    P = []
    for H in data:
        node_degrees = compute_node_degrees(H)  
        hyperedge_degrees = compute_hyperedge_degrees(H)  
        P1 = compute_first_order_transition_probabilities(
            H, node_degrees, hyperedge_degrees  
        )
        P.append(P1)
    return np.array(P)



def run_experiment(seed, X_selected_train, y_selected_train, 
                   X_selected_val, y_selected_val, X_test_filenames):

    # 1. Data shuffling and preprocessing
    X_train, y_train_bal = shuffle(X_selected_train, y_selected_train, random_state=seed)
    X_val, y_val_bal = shuffle(X_selected_val, y_selected_val, random_state=seed)
    
    P_train = generate_P(X_train)  
    P_val = generate_P(X_val)
    input_shape = (11, 11)
    
    #learning rate
    all_lr_results = []
    lrs = [0.001,0.0001, 0.003,0.0003]  
    
    for lr in lrs:
        tf.random.set_seed(seed)
        model = build_optimized_model(input_shape, lr) 

        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train_bal), y=y_train_bal
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        checkpoint_path = f'best_model_seed_{seed}_lr_{lr}.h5'
        callbacks = [
            ModelCheckpoint(
                filepath=checkpoint_path, 
                monitor='val_f1_score',  
                save_best_only=True, 
                save_weights_only=False, 
                mode='max',  
                verbose=1
            )
        ]

        # Train the model and capture the history
        history = model.fit(
            P_train, y_train_bal,
            validation_data=(P_val, y_val_bal),
            batch_size=32, epochs=100,
            shuffle=True, class_weight=class_weight_dict,
            callbacks=callbacks
        )

    
        val_f1_scores = history.history['val_f1_score']
        best_f1 = max(val_f1_scores)  
        best_epoch = val_f1_scores.index(best_f1) 

        val_results = {
            'lr': lr,  
            'best_epoch': best_epoch + 1,  
            'f1': best_f1,
            'auc': history.history['val_auc'][best_epoch],
            'precision': history.history['val_precision'][best_epoch],
            'recall': history.history['val_recall'][best_epoch]
        }

        test_results = {}
        best_model = tf.keras.models.load_model(
            checkpoint_path,  
            custom_objects={
                'AdaptiveFocalLoss': AdaptiveFocalLoss,
                'DynamicConv1D': DynamicConv1D,
                'Module': Module,
                'F1Score': F1Score
            }
        )
        
        for file in X_test_filenames:
            if not isinstance(file, str):
                print(f"Warning: Invalid file path (not a string): {file}, skipped")
                continue
            
            # Extract chromosome names
            match = re.search(r"chr\d+", file)
            chr_name = match.group() if match else f"file_{os.path.basename(file)}"
            
           
            try:
                X_test, _, y_test, _ = process_files_to_arrays([file]) 
                P_test = generate_P(X_test)
            except Exception as e:
                print(f"Warning: Failed to load file {file}: {e}, skipped")
                continue
            
          
            y_pred_prob = best_model.predict(P_test, verbose=0).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            auc = tf.keras.metrics.AUC()(y_test, y_pred_prob).numpy()
            precision = tf.keras.metrics.Precision()(y_test, y_pred).numpy()
            recall = tf.keras.metrics.Recall()(y_test, y_pred).numpy()
            f1 = F1Score()(y_test, y_pred).numpy()
            
            test_results[chr_name] = {
                'lr': lr,  
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        all_lr_results.append({
            'val': val_results,  
            'test': test_results
        })
    return all_lr_results
    

def print_class_distribution(y, description=""):
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    
    print(f"{description} sample distribution:")
    print(f"Positive class (1): {class_distribution.get(1, 0)}")
    print(f"Negative class (0): {class_distribution.get(0, 0)}")
    print(f"Total number of samples: {sum(counts)}")
    print(f"Ratio of positive and negative samples: {class_distribution.get(1, 0) / sum(counts):.2%} positive class, {class_distribution.get(0, 0) / sum(counts):.2%} negative class\n")
def process_and_merge_chromosomes(X, mid_indices, y, sample_counts, select):
    """
    Split data by chromosome, perform sampling, and merge the results
    Parameters:
        X (np.ndarray): Original feature data (shape: [total number of samples, feature dimensions...])
        mid_indices (np.ndarray): Intermediate row indices (shape: [total number of samples, ...])
        y (np.ndarray): Original labels (shape: [total number of samples,])
        sample_counts (list): List of sample counts for each chromosome (length = number of chromosomes)
        select (callable): Sampling function, which must satisfy select_func(X_chr, mid_chr, y_chr) -> (X_sel, mid_sel, y_sel)
    Returns:
        X_selected (np.ndarray): Merged sampled features
        mid_selected (np.ndarray): Merged intermediate row indices
        y_selected (np.ndarray): Merged sampled labels
    """
    X_selected_list = []
    mid_selected_list = []
    y_selected_list = []
    
    start_idx = 0
    for chr_idx, count in enumerate(sample_counts):
        end_idx = start_idx + count
        
      
        X_chr = X[start_idx:end_idx]
        mid_chr = mid_indices[start_idx:end_idx]
        y_chr = y[start_idx:end_idx]
        
        X_sel, mid_sel, y_sel = select(X_chr, mid_chr, y_chr,5)
        
        X_selected_list.append(X_sel)
        mid_selected_list.append(mid_sel)
        y_selected_list.append(y_sel)
        
        pos = np.sum(y_sel == 1)
        neg = np.sum(y_sel == 0)
        print(f"Chromosome {chr_idx+1} processed | positive samples: {pos} | negative samples: {neg} | total samples: {len(y_sel)}")
        
        start_idx = end_idx  
    
  
    X_selected = np.concatenate(X_selected_list, axis=0)
    mid_selected = np.concatenate(mid_selected_list, axis=0)
    y_selected = np.concatenate(y_selected_list, axis=0)

    total_pos = np.sum(y_selected == 1)
    total_samples = len(y_selected)
    print(f"\nAll chromosomes merged | feature shape: {X_selected.shape} | positive sample ratio: {total_pos}/{total_samples}")
    
    return X_selected, mid_selected, y_selected



dir1 = '/mnt/sdb/liukaihua/Article/model/model_data/'
dir2 = '25_sub_matrix.txt'
chr_list = [f"chr{i}" for i in range(1, 23)]
def data(chr_list):
    file = []
    X_train_filenames = []
    X_test_filenames = []
    X_val_filenames = []
    for chr in chr_list:
        if chr == "chr3" or chr == "chr18":
            continue
        f1 = os.path.join(dir1, f"{chr}_{dir2}")
        file.append(f1)
       
    # print(file)
    X_train_filenames = file[:11]
    X_val_filenames = file[11:17]
    X_test_filenames = file[17:21]
    return X_test_filenames,X_train_filenames,X_val_filenames

X_test_filenames,X_train_filenames,X_val_filenames= data(chr_list)
print("Number of training set file paths:", X_train_filenames)
print("Number of validation set file paths:", X_val_filenames)
print("Number of test set file paths:", X_test_filenames)
  
    
X_train, middle_row_indices_train, y_train,sample_counts_train = process_files_to_arrays(X_train_filenames)
X_val, middle_row_indices_val, y_val,sample_counts_val = process_files_to_arrays(X_val_filenames)
# X_test, middle_row_indices_test, y_test,sample_counts_test = process_files_to_arrays(X_test_filenames)
print_class_distribution(y_train, description="Train")
print_class_distribution(y_val, description="Val")


def main():
  
    X_selected_train, mid_selected_train, y_selected_train = process_and_merge_chromosomes(
        X_train, middle_row_indices_train, y_train, sample_counts_train, select
    )  
    X_selected_val, mid_selected_val, y_selected_val = process_and_merge_chromosomes(
        X_val, middle_row_indices_val, y_val, sample_counts_val, select
    ) 
    

    # -------------------------- experimental configuration--------------------------
    base_seeds = [42]  
    repeats_per_seed = 5  # Number of repetitions for each base seed
    output_root = 'multi_seed_lr_experiment_results' 
    os.makedirs(output_root, exist_ok=True)
    
   
    val_metrics_all = {
        'base_seed': [], 'repeat': [], 'lr': [],  
        'auc': [], 'precision': [], 'recall': [], 'f1': []
    }
    test_metrics_all = {}  
    # -------------------------- run the experiment --------------------------
    total_experiments = len(base_seeds) * repeats_per_seed
    current_exp = 1
    
    for base_seed in base_seeds:
        print(f"\n===== Start {repeats_per_seed} repeated experiments for base seed {base_seed} =====")
        for repeat in range(repeats_per_seed):
            exp_seed = base_seed * 100 + repeat 
            print(f"Experiment {current_exp}/{total_experiments} | Base seed: {base_seed} | Repetition: {repeat+1}/{repeats_per_seed} | Sub-seed: {exp_seed}")
            
            all_lr_results = run_experiment(
                seed=exp_seed,
                X_selected_train=X_selected_train,
                y_selected_train=y_selected_train,
                X_selected_val=X_selected_val,
                y_selected_val=y_selected_val,
                X_test_filenames=X_test_filenames
            )
            for lr_result in all_lr_results:
                lr = lr_result['val']['lr']  
                
                val_res = lr_result['val']
                val_metrics_all['base_seed'].append(base_seed)
                val_metrics_all['repeat'].append(repeat+1)
                val_metrics_all['lr'].append(lr)
                val_metrics_all['auc'].append(val_res['auc'])
                val_metrics_all['precision'].append(val_res['precision'])
                val_metrics_all['recall'].append(val_res['recall'])
                val_metrics_all['f1'].append(val_res['f1'])
                test_res = lr_result['test']
                for chr_name, metrics in test_res.items():
                    if chr_name not in test_metrics_all:
                        test_metrics_all[chr_name] = {
                            'base_seed': [], 'repeat': [], 'lr': [],
                            'auc': [], 'precision': [], 'recall': [], 'f1': []
                        }
                    test_metrics_all[chr_name]['base_seed'].append(base_seed)
                    test_metrics_all[chr_name]['repeat'].append(repeat+1)
                    test_metrics_all[chr_name]['lr'].append(lr)
                    test_metrics_all[chr_name]['auc'].append(metrics['auc'])
                    test_metrics_all[chr_name]['precision'].append(metrics['precision'])
                    test_metrics_all[chr_name]['recall'].append(metrics['recall'])
                    test_metrics_all[chr_name]['f1'].append(metrics['f1'])
            
            current_exp += 1

    # -------------------------- save the results --------------------------

    val_df = pd.DataFrame(val_metrics_all)
    for lr in val_df['lr'].unique():
        lr_dir = os.path.join(output_root, f'lr_{lr}')
        os.makedirs(lr_dir, exist_ok=True)

        val_lr_df = val_df[val_df['lr'] == lr]
        val_lr_df.to_csv(os.path.join(lr_dir, 'validation_metrics_details.csv'), index=False)
        val_summary = pd.DataFrame({
            'metric': ['auc', 'precision', 'recall', 'f1'],
            'overall_mean±sd': [
                f"{val_lr_df['auc'].mean():.3f}±{val_lr_df['auc'].std(ddof=1):.3f}",
                f"{val_lr_df['precision'].mean():.3f}±{val_lr_df['precision'].std(ddof=1):.3f}",
                f"{val_lr_df['recall'].mean():.3f}±{val_lr_df['recall'].std(ddof=1):.3f}",
                f"{val_lr_df['f1'].mean():.3f}±{val_lr_df['f1'].std(ddof=1):.3f}"
            ]
        })
        for base_seed in base_seeds:
            seed_subset = val_lr_df[val_lr_df['base_seed'] == base_seed]
            val_summary[f'seed_{base_seed}_mean±sd'] = [
                f"{seed_subset['auc'].mean():.3f}±{seed_subset['auc'].std(ddof=1):.3f}",
                f"{seed_subset['precision'].mean():.3f}±{seed_subset['precision'].std(ddof=1):.3f}",
                f"{seed_subset['recall'].mean():.3f}±{seed_subset['recall'].std(ddof=1):.3f}",
                f"{seed_subset['f1'].mean():.3f}±{seed_subset['f1'].std(ddof=1):.3f}"
            ]
        val_summary.to_csv(os.path.join(lr_dir, 'validation_metrics_summary.csv'), index=False)

    for chr_name, metrics in test_metrics_all.items():
        test_df = pd.DataFrame(metrics)
        for lr in test_df['lr'].unique():
            lr_dir = os.path.join(output_root, f'lr_{lr}')
            os.makedirs(lr_dir, exist_ok=True)
            test_lr_df = test_df[test_df['lr'] == lr]
            test_lr_df.to_csv(
                os.path.join(lr_dir, f'test_{chr_name}_metrics_details.csv'),
                index=False
            )
            test_summary = pd.DataFrame({
                'metric': ['auc', 'precision', 'recall', 'f1'],
                'overall_mean±sd': [
                    f"{test_lr_df['auc'].mean():.3f}±{test_lr_df['auc'].std(ddof=1):.3f}",
                    f"{test_lr_df['precision'].mean():.3f}±{test_lr_df['precision'].std(ddof=1):.3f}",
                    f"{test_lr_df['recall'].mean():.3f}±{test_lr_df['recall'].std(ddof=1):.3f}",
                    f"{test_lr_df['f1'].mean():.3f}±{test_lr_df['f1'].std(ddof=1):.3f}"
                ]
            })
            for base_seed in base_seeds:
                seed_subset = test_lr_df[test_lr_df['base_seed'] == base_seed]
                test_summary[f'seed_{base_seed}_mean±sd'] = [
                    f"{seed_subset['auc'].mean():.3f}±{seed_subset['auc'].std(ddof=1):.3f}",
                    f"{seed_subset['precision'].mean():.3f}±{seed_subset['precision'].std(ddof=1):.3f}",
                    f"{seed_subset['recall'].mean():.3f}±{seed_subset['recall'].std(ddof=1):.3f}",
                    f"{seed_subset['f1'].mean():.3f}±{seed_subset['f1'].std(ddof=1):.3f}"
                ]
            test_summary.to_csv(
                os.path.join(lr_dir, f'test_{chr_name}_metrics_summary.csv'),
                index=False
            )
    
    print(f"\nAll results have been classified by learning rate and saved to the {output_root} folder")

if __name__ == "__main__":
    main()
