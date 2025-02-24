'''
This code creates a class to simulate the LIF motor neuron 

Ryan Konno
'''

class NeuronSimulator():
    def __init__(self, params):

        self.params = params

        self.G = params['G']
        self.I0 = params['I0']
        
        # self.fsamp_exp = params['fsamp'] # experimental data sample freq (same idxs as excitation)

        # print('Simulator initialized...')

    def currentFun(self, t): 
        '''
        Function to compute the current given some excitation

        The parameters for this function self.G, self.I0 are determined previously through based on the input functions
        '''
        return self.I0 + self.G * self.excitation[int(t * self.fsamp)]

    def simulateLIF(self, neuron_model, excitation, fsamp = 2048):
        '''
        Function to call the MU solver
        '''
        # print('Simulating neuron...')

        self.excitation = excitation
        self.fsamp = fsamp

        t_vec, V_vec, t_fire_vec = neuron_model.solveNeuron(self.params['t_end'], self.currentFun)
        
        return t_vec, V_vec, t_fire_vec

