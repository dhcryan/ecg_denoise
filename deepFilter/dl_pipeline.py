# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import deepFilter.dl_models as models  # PyTorch로 변환된 모델들

# # Custom loss functions (PyTorch version)
# def ssd_loss(y_true, y_pred):
#     return torch.sum((y_pred - y_true) ** 2, dim=-2)

# def combined_ssd_mse_loss(y_true, y_pred):
#     return torch.mean((y_true - y_pred) ** 2, dim=-2) * 500 + torch.sum((y_true - y_pred) ** 2, dim=-2)

# def combined_ssd_mad_loss(y_true, y_pred):
#     return torch.max((y_true - y_pred) ** 2, dim=-2).values * 50 + torch.sum((y_true - y_pred) ** 2, dim=-2)

# def sad_loss(y_true, y_pred):
#     return torch.sum(torch.sqrt((y_pred - y_true) ** 2), dim=-2)

# def mad_loss(y_true, y_pred):
#     return torch.max((y_pred - y_true) ** 2, dim=-2).values

# def train_dl(Dataset, experiment):

#     print(f'Deep Learning pipeline: Training the model for exp {experiment}')

#     X_train, y_train, X_test, y_test = Dataset

#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)

#     # Load the model based on experiment
#     if experiment == 'FCN-DAE':
#         model = models.FCN_DAE()
#         model_label = 'FCN_DAE'
#     elif experiment == 'DRNN':
#         model = models.DRRN_denoising()
#         model_label = 'DRNN'
#     elif experiment == 'Vanilla L':
#         model = models.DeepFilterVanillaLinear()
#         model_label = 'Vanilla_L'
#     elif experiment == 'Vanilla NL':
#         model = models.DeepFilterVanillaNLinear()
#         model_label = 'Vanilla_NL'
#     elif experiment == 'Multibranch LANL':
#         model = models.DeepFilterILANL()
#         model_label = 'Multibranch_LANL'
#     elif experiment == 'Multibranch LANLD':
#         model = models.DeepFilterModelILANLDilated()
#         model_label = 'Multibranch_LANLD'
#     elif experiment == 'Transformer_DAE':
#         model = models.TransformerDAE()
#         model_label = 'Transformer_DAE'
#     elif experiment == 'Transformer_FDAE':
#         model = models.TransformerFDAE()
#         model_label = 'Transformer_FDAE'
#     else:
#         raise ValueError("Unknown experiment type!")

#     # GPU 또는 CPU로 모델을 이동
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = model.to(device)

#     print(f'\n{model_label}\n')
    
#     # Hyperparameters
#     epochs = 100
#     batch_size = 128
#     lr = 1e-3
#     minimum_lr = 1e-10

#     # Select loss function
#     if experiment == 'DRNN':
#         criterion = nn.MSELoss()
#     elif experiment == 'FCN-DAE':
#         criterion = ssd_loss
#     else:
#         criterion = combined_ssd_mad_loss

#     # Optimizer
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     # Scheduler
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=minimum_lr, verbose=True)

#     # Prepare data loaders
#     train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
#     val_data = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())

#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_data, batch_size=batch_size)

#     # Training loop
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for i, (inputs, labels) in enumerate(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device)

#             # 입력 차원을 (batch_size, channels, sequence_length) 형태로 변환
#             inputs = inputs.permute(0, 2, 1)

#             optimizer.zero_grad()

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         val_loss = 0.0
#         model.eval()
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)

#                 # 입력 차원을 (batch_size, channels, sequence_length) 형태로 변환
#                 inputs = inputs.permute(0, 2, 1)

#                 outputs = model(inputs)
#                 val_loss += criterion(outputs, labels).item()

#         print(f'Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')

#         scheduler.step(val_loss)

#     # Save model weights
#     torch.save(model.state_dict(), f'{model_label}_weights.best.pth')



# def test_dl(Dataset, experiment):

#     print('Deep Learning pipeline: Testing the model')

#     X_train, y_train, X_test, y_test = Dataset
#     batch_size = 32

#     # Load the model based on experiment
#     if experiment == 'FCN-DAE':
#         model = models.FCN_DAE()
#         model_label = 'FCN_DAE'
#     elif experiment == 'DRNN':
#         model = models.DRRN_denoising()
#         model_label = 'DRNN'
#     elif experiment == 'Vanilla L':
#         model = models.DeepFilterVanillaLinear()
#         model_label = 'Vanilla_L'
#     elif experiment == 'Vanilla NL':
#         model = models.DeepFilterVanillaNLinear()
#         model_label = 'Vanilla_NL'
#     elif experiment == 'Multibranch LANL':
#         model = models.DeepFilterILANL()
#         model_label = 'Multibranch_LANL'
#     elif experiment == 'Multibranch LANLD':
#         model = models.DeepFilterModelILANLDilated()
#         model_label = 'Multibranch_LANLD'
#     elif experiment == 'Transformer_DAE':
#         model = models.TransformerDAE()
#         model_label = 'Transformer_DAE'
#     elif experiment == 'Transformer_FDAE':
#         model = models.TransformerFDAE()
#         model_label = 'Transformer_FDAE'
#     else:
#         raise ValueError("Unknown experiment type!")

#     # 모델을 CPU 또는 GPU로 이동
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = model.to(device)

#     # Load model weights
#     model.load_state_dict(torch.load(f'{model_label}_weights.best.hdf5'))

#     # Prepare data loader for testing
#     test_data = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
#     test_loader = DataLoader(test_data, batch_size=batch_size)

#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for inputs, _ in test_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             predictions.append(outputs.cpu().numpy())

#     return X_test, y_test, predictions

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import losses
from sklearn.model_selection import train_test_split
import tensorflow as tf 

import deepFilter.dl_models as models
# Custom loss SSD
def ssd_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-2)

# Combined loss SSD + MSE
def combined_ssd_mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-2) * 500 + K.sum(K.square(y_true - y_pred), axis=-2)

def combined_ssd_mad_loss(y_true, y_pred):
    return K.max(K.square(y_true - y_pred), axis=-2) * 50 + K.sum(K.square(y_true - y_pred), axis=-2)

# Custom loss SAD
def sad_loss(y_true, y_pred):
    return K.sum(K.sqrt(K.square(y_pred - y_true)), axis=-2)

# Custom loss MAD
def mad_loss(y_true, y_pred):
    return K.max(K.square(y_pred - y_true), axis=-2)


def train_dl(Dataset, experiment):

    print('Deep Learning pipeline: Training the model for exp ' + str(experiment))

    [X_train, y_train, X_test, y_test] = Dataset

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)

    # ==================
    # LOAD THE DL MODEL
    # ==================

    if experiment == 'FCN-DAE':
        # FCN_DAE
        model = models.FCN_DAE()
        model_label = 'FCN_DAE'

    if experiment == 'DRNN':
        # DRNN
        model = models.DRRN_denoising()
        model_label = 'DRNN'

    if experiment == 'Vanilla L':
        # Vanilla CNN linear
        model = models.deep_filter_vanilla_linear()
        model_label = 'Vanilla_L'

    if experiment == 'Vanilla NL':
        # Vanilla CNN non linear
        model = models.deep_filter_vanilla_Nlinear()
        model_label = 'Vanilla_NL'    

    if experiment == 'Multibranch LANL':
        # Multibranch linear and non linear
        model = models.deep_filter_I_LANL()
        model_label = 'Multibranch_LANL'

    if experiment == 'Multibranch LANLD':
        # Inception-like linear and non linear dilated
        model = models.deep_filter_model_I_LANL_dilated()
        model_label = 'Multibranch_LANLD'
        
    if experiment == 'Transformer_DAE':
        # Transformer_DAE
        model = models.Transformer_DAE()
        model_label = 'Transformer_DAE'
    
    if experiment == 'Transformer_FDAE':
        # Transformer_FDAE
        model = models.Transformer_FDAE()
        model_label = 'Transformer_FDAE'
    
    if experiment == 'MemoryTransformer_DAE':
        # MemoryTransformer_DAE
        model = models.MemoryTransformer_DAE()
        model_label = 'MemoryTransformer_DAE'

    print('\n ' + model_label + '\n ')

    model.summary()

    epochs = int(1e2)  # 100000
    # epochs = 100
    batch_size = 128
    lr = 1e-3
    # lr = 1e-4
    minimum_lr = 1e-10


    # Loss function selection according to method implementation
    if experiment == 'DRNN':
        criterion = keras.losses.mean_squared_error

    elif experiment == 'FCN-DAE':
        criterion = ssd_loss

    else:
        criterion = combined_ssd_mad_loss


    model.compile(loss=criterion,
                  optimizer=tf.keras.optimizers.Adam(lr=lr),
                  metrics=[losses.mean_squared_error, losses.mean_absolute_error, ssd_loss, mad_loss])

    # Keras Callbacks

    # checkpoint
    model_filepath = model_label + '_weights.best.hdf5'

    checkpoint = ModelCheckpoint(model_filepath,
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',  # on acc has to go max
                                 save_weights_only=True)

    reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                  factor=0.5,
                                  min_delta=0.05,
                                  mode='min',  # on acc has to go max
                                  patience=2,
                                  min_lr=minimum_lr,
                                  verbose=1)

    early_stop = EarlyStopping(monitor="val_loss",  # "val_loss"
                               min_delta=0.05,
                               mode='min',  # on acc has to go max
                               patience=10,
                               verbose=1)

    tb_log_dir = './runs/' + model_label

    tboard = TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                         write_graph=False, write_grads=False,
                         write_images=False, embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

    # To run the tensor board
    # tensorboard --logdir=./runs

    # GPU
    model.fit(x=X_train, y=y_train,
              validation_data=(X_val, y_val),
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[early_stop,
                         reduce_lr,
                         checkpoint,
                         tboard])

    K.clear_session()



def test_dl(Dataset, experiment):

    print('Deep Learning pipeline: Testing the model')

    [train_set, train_set_GT, X_test, y_test] = Dataset

    batch_size = 32

    # ==================
    # LOAD THE DL MODEL
    # ==================

    if experiment == 'FCN-DAE':
        # FCN_DAE
        model = models.FCN_DAE()
        model_label = 'FCN_DAE'

    if experiment == 'DRNN':
        # DRNN
        model = models.DRRN_denoising()
        model_label = 'DRNN'

    if experiment == 'Vanilla L':
        # Vanilla CNN linear
        model = models.deep_filter_vanilla_linear()
        model_label = 'Vanilla_L'

    if experiment == 'Vanilla NL':
        # Vanilla CNN non linear
        model = models.deep_filter_vanilla_Nlinear()
        model_label = 'Vanilla_NL'    

    if experiment == 'Multibranch LANL':
        # Multibranch linear and non linear
        model = models.deep_filter_I_LANL()
        model_label = 'Multibranch_LANL'

    if experiment == 'Multibranch LANLD':
        # Inception-like linear and non linear dilated
        model = models.deep_filter_model_I_LANL_dilated()
        model_label = 'Multibranch_LANLD'
        
    if experiment == 'Transformer_DAE':
        # Transformer_DAE
        model = models.Transformer_DAE()
        model_label = 'Transformer_DAE'
    
    if experiment == 'Transformer_FDAE':
        # Transformer_FDAE
        model = models.Transformer_FDAE()
        model_label = 'Transformer_FDAE'
    
    if experiment == 'MemoryTransformer_DAE':
        # MemoryTransformer_DAE
        model = models.MemoryTransformer_DAE()
        model_label = 'MemoryTransformer_DAE'
    
    print('\n ' + model_label + '\n ')

    model.summary()

    # Loss function selection according to method implementation
    if experiment == 'DRNN':
        criterion = 'mse'

    elif experiment == 'FCN-DAE':
        criterion = ssd_loss

    else:
        criterion = combined_ssd_mad_loss

    model.compile(loss=criterion,
                  optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  metrics=[losses.mean_squared_error, losses.mean_absolute_error, ssd_loss, mad_loss])

    # checkpoint
    model_filepath = model_label + '_weights.best.hdf5'
    # load weights
    model.load_weights(model_filepath)

    # Test score
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)


    K.clear_session()

    return [X_test, y_test, y_pred]