import numpy as np
import json

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    expx = np.exp(x)
    return expx/np.sum(expx, axis=-1, keepdims=True)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
    
def tanh(x):
    return sigmoid(2*x)*2-1
    
def relu(x):
    return (x+np.abs(x))/2    

activations = {
    "tanh": tanh,
    "sigmoid":  sigmoid,
    "softmax":  softmax,
    "relu":     relu
}

class Dense(object):
    
    NWeights = 2
    
    def __init__(self, in_width, out_width, activation="softmax"):
        self.InWidth = in_width
        self.OutWidth = out_width
        self.W = None
        self.B = None
        self.Activation = activations[activation]

    def set_weights(self, w, b):
        self.W = w
        self.B = b
        
    def compute(self, x):
        return self.Activation(np.dot(x, self.W) + self.B)

class LSTM(object):

    NWeights = 3
    
    def __init__(self, in_width, out_width, return_sequences=False, activation="tanh"):
        self.InWidth = in_width
        self.OutWidth = out_width
        self.XW = None
        self.SW = None
        self.B = None
        self.ReturnSequences = False
        self.Activation = activations[activation]
        
    def set_weights(self, xw, sw, b):
        self.XW = xw
        self.SW = sw
        self.B = b
        
    def compute(self, batch):
        # batch: (mb_size, time_length, in_width)
        mb_size, time_length, in_width = batch.shape
        assert in_width == self.InWidth
        state_c = np.zeros((mb_size, self.OutWidth))
        state_h = np.zeros((mb_size, self.OutWidth))
        Y = np.empty((mb_size, time_length, self.OutWidth))
        for t in range(time_length):
            xt = batch[:,t,:]
            z = np.dot(state_h, self.SW) + np.dot(xt, self.XW) + self.B
            f = sigmoid(z[:,self.OutWidth:self.OutWidth*2])
            I = sigmoid(z[:,:self.OutWidth])
            c = np.tanh(z[:,self.OutWidth*2:self.OutWidth*3])
            o = sigmoid(z[:,self.OutWidth*3:])
            
            state_c = state_c*f + I*c
            state_h = self.Activation(state_c)*o
            
            Y[:,t,:] = state_h
        if self.ReturnSequences:
            return Y
        else:
            return Y[:,-1,:]

class Model(object):
    
    def __init__(self, layers):
        self.Layers = layers
        
    def set_weights(self, weights):
        for l in self.Layers:
            n = l.NWeights
            l.set_weights(*weights[:n])
            weights = weights[n:]

    def compute(self, x):
        for l in self.Layers:
            x = l.compute(x)
        return x
    
    @staticmethod
    def save_keras_model(model, file_prefix):
        open(file_prefix+"_desc.json", "w").write(model.to_json())
        np.savez(file_prefix+"_weights.npz", *model.get_weights())

    @staticmethod
    def from_saved(file_prefix):
        desc = json.load(open(file_prefix+"_desc.json", "r"))
        weights = np.load(file_prefix+"_weights.npz")
        weights = [weights[name] for name in weights.files]
        return Model.create_from_desc_and_weights(desc, weights)
        
    @staticmethod
    def create_from_desc_and_weights(desc, weights):
        if isinstance(desc, str):
            desc = json.loads(desc)
        layers = []
        config = desc["config"]
        shape = None
        for ldesc in config["layers"]:
            cname = ldesc["class_name"]
            lcfg = ldesc["config"]
            if cname == "InputLayer":
                shape = lcfg["batch_input_shape"]
            elif cname == "LSTM":
                width = lcfg["units"]
                activation = lcfg["activation"]
                sequences = lcfg["return_sequences"]
                layers.append(LSTM(shape[-1], width, activation=activation, return_sequences=sequences))
                shape = (shape[0], width) if not sequences else (shape[0], shape[1], width)
            elif cname == "Dense":
                width = lcfg["units"]
                activation = lcfg["activation"]
                layers.append(Dense(shape[-1], width, activation=activation))
                shape = [width]

        model = Model(layers)
        model.set_weights(weights)
        return model
                

        