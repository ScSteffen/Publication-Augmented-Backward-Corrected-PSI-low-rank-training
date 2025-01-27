import torch
import torch.nn as nn
import csv

def create_layer(layer):
    if layer['type'] == 'dense':
        return DenseLayer(layer['dims'][0], layer['dims'][1])
    if layer['type'] == 'PSI_dynamical_low_rank':
        return PSI_LowRankLayer(layer['dims'][0], layer['dims'][1], layer['rank'])
    if layer['type'] == 'PSI_Backward_dynamical_low_rank':
        return PSI_Backward_LowRankLayer(layer['dims'][0], layer['dims'][1], layer['rank'])
    if layer['type'] == 'PSI_Augmented_Backward_dynamical_low_rank':
        return PSI_Augmented_Backward_LowRankLayer(layer['dims'][0], layer['dims'][1], layer['rank'], layer['tol'])


# Define standard layer
class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        """Constructs a dense layer of the form W*x + b, where W is the weigh matrix and b is the bias vector
        Args:
            input_size: input dimension of weight W
            output_size: output dimension of weight W, dimension of bias b
        """
        # construct parent class nn.Module
        super(DenseLayer, self).__init__()
        # define weights as trainable parameter
        self.W = nn.Parameter(torch.randn(input_size, output_size))
        # define bias as trainable parameter
        self.bias = nn.Parameter(torch.randn(output_size))

    def forward(self, x, mode='dense'):
        """Returns the output of the layer. The formula implemented is output = W*x + bias.
        Args:
            x: input to layer
        Returns: 
            output of layer
        """
        out = torch.matmul(x, self.W)
        return out + self.bias

    def step(self, learning_rate, mode='dense'):
        """Performs a steepest descend training update on weights and biases 
        Args:
            learning_rate: learning rate for training
        """
        self.W.data = self.W - learning_rate * self.W.grad
        self.bias.data = self.bias - learning_rate * self.bias.grad

    def set_all_zero(self):
        if self.W.grad is not None: self.W.grad.zero_()
        if self.bias.grad is not None: self.bias.grad.zero_()

    def write(self, file_name, use_txt=True):
        """Writes all weight matrices 
        Args:
            file_name: name of the file format in which weights are stored
        """
        # save as pth
        torch.save(self.W, file_name + "_W.pth")
        torch.save(self.bias, file_name + "_b.pth")

        if use_txt:
            with open(file_name + "_W.txt", 'w') as file:
                for row in self.W.data.T:
                    row_str = '\t'.join(map(str, row.tolist()))
                    file.write(row_str + '\n')

            with open(file_name + "_b.txt", 'w') as file:
                bias_str = '\t'.join(map(str, self.bias.data.tolist()))
                file.write(bias_str)



#################################################################################################################################################
### Define low-rank layer with Projector Splitting Integrator ###################################################################################
#################################################################################################################################################

class PSI_LowRankLayer(nn.Module):

    def __init__(self, input_size, output_size, rank):
        """Constructs a low-rank layer of the form U*S*V'*x + b, where 
           U, S, V represent the facorized weight W and b is the bias vector
        Args:
            input_size: input dimension of weight W
            output_size: output dimension of weight W, dimension of bias b
        """
        # construct parent class nn.Module
        super(PSI_LowRankLayer, self).__init__()

        self.rank = rank

        # initializes factorized weight
        self.U = nn.Parameter(torch.randn(input_size, rank), requires_grad = True)
        self.S = nn.Parameter(torch.randn(rank, rank),  requires_grad = True)
        self.V = nn.Parameter(torch.randn(output_size, rank),  requires_grad = True)

        # ensure that U and V are orthonormal
        self.U.data, _ = torch.linalg.qr(self.U, 'reduced')
        self.V.data, _ = torch.linalg.qr(self.V, 'reduced')

        # initializes combined weight
        self.K = nn.Parameter(torch.randn(input_size, rank), requires_grad = True)
        self.K.data = torch.matmul(self.U, self.S)
        #self.K.retain_grad()

        self.L = nn.Parameter(torch.randn(output_size, rank), requires_grad = True)
        self.L.data = torch.matmul(self.V, self.S.T)
        #self.L.retain_grad()

        # initialize bias
        self.bias = nn.Parameter(torch.randn(output_size))


    def forward(self, x, dlrt_step):
        """Returns the output of the layer. The formula implemented is output = U*S*V'*x + bias.
        Args:
            x: input to layer
        Returns: 
            output of layer
        """
        if dlrt_step == 'K':
            xUS = torch.matmul(x, self.K)
            out = torch.matmul(xUS, self.V.T)

        elif dlrt_step == 'S':
            xU = torch.matmul(x, self.U )
            xUS = torch.matmul(xU, self.S)
            out = torch.matmul(xUS, self.V.T)

        elif dlrt_step == 'L':
            xU = torch.matmul(x, self.U )
            out = torch.matmul(xU, self.L.T)

        elif dlrt_step == 'test':
            xU = torch.matmul(x, self.U )
            xUS = torch.matmul(xU, self.S)
            out = torch.matmul(xUS, self.V.T)

        #print("out.shape:", out.shape)
        return out + self.bias

    @torch.no_grad()
    def step(self, learning_rate, dlrt_step):
        """Performs a steepest descend training update on specified low-rank factors
        Args:
            learning_rate: learning rate for training
            dlrt_step: sepcifies step that is taken. Can be 'K', 'L' or 'S'
        """          
        # perform K-step
        if dlrt_step == 'K':
            self.K.data = self.K - learning_rate * self.K.grad
            self.U.data , self.S.data = torch.linalg.qr(self.K, 'reduced')

        # perform S-step
        elif dlrt_step == 'S':
            self.S.data = self.S + learning_rate * self.S.grad
            self.L.data = torch.matmul(self.V, self.S.T)  

        # perform L-step
        elif dlrt_step == 'L':
            self.L.data = self.L - learning_rate * self.L.grad
            self.V.data , tmps = torch.linalg.qr(self.L, 'reduced')
            self.S.data = tmps.T
            self.K.data = torch.matmul(self.U, self.S)

            # update bias
            self.bias.data = self.bias - learning_rate * self.bias.grad

        else:
            print("Wrong step defined: ", dlrt_step)

    def set_all_zero(self):
        if self.S.grad is not None: self.S.grad.zero_()
        if self.U.grad is not None: self.U.grad.zero_()
        if self.V.grad is not None: self.V.grad.zero_()
        if self.K.grad is not None: self.K.grad.zero_()
        if self.L.grad is not None: self.L.grad.zero_()
        if self.bias.grad is not None: self.bias.grad.zero_()

    def get_layer_rank(self):
        return self.rank
    
    def write(self, file_name, use_txt=True):
        """Writes all weight matrices 
        Args:
            file_name: name of the file format in which weights are stored
        """
        # save as pth
        torch.save(self.U, file_name + "_U.pth")
        torch.save(self.S, file_name + "_S.pth")
        torch.save(self.V, file_name + "_V.pth")
        torch.save(self.bias, file_name + "_b.pth")

        if use_txt:
            with open(file_name + "_U.txt", 'w') as file:
                for row in self.U.data.T:
                    row_str = '\t'.join(map(str, row.tolist()))
                    file.write(row_str + '\n')
            with open(file_name + "_S.txt", 'w') as file:
                for row in self.S.data.T:
                    row_str = '\t'.join(map(str, row.tolist()))
                    file.write(row_str + '\n')
            with open(file_name + "_V.txt", 'w') as file:
                for row in self.V.data.T:
                    row_str = '\t'.join(map(str, row.tolist()))
                    file.write(row_str + '\n')
            with open(file_name + "_b.txt", 'w') as file:
                bias_str = '\t'.join(map(str, self.bias.data.tolist()))
                file.write(bias_str)


#################################################################################################################################################
### Define low-rank layer with Backward Corrected Projector Splitting Integrator ################################################################
#################################################################################################################################################

class PSI_Backward_LowRankLayer(nn.Module):

    def __init__(self, input_size, output_size, rank):
        """Constructs a low-rank layer of the form U*S*V'*x + b, where 
           U, S, V represent the facorized weight W and b is the bias vector
        Args:
            input_size: input dimension of weight W
            output_size: output dimension of weight W, dimension of bias b
        """
        # construct parent class nn.Module
        super(PSI_Backward_LowRankLayer, self).__init__()

        self.rank = rank

        # initializes factorized weight
        self.U = nn.Parameter(torch.randn(input_size, rank), requires_grad = True)
        self.S = nn.Parameter(torch.randn(rank, rank),  requires_grad = True)
        self.V = nn.Parameter(torch.randn(output_size, rank),  requires_grad = True)
        self.U0 = nn.Parameter(torch.randn(input_size, rank), requires_grad = False)

        # ensure that U and V are orthonormal
        self.U.data, _ = torch.linalg.qr(self.U, 'reduced')
        self.V.data, _ = torch.linalg.qr(self.V, 'reduced')

        # initializes combined weight
        self.K = nn.Parameter(torch.randn(input_size, rank), requires_grad = True)
        self.K.data = torch.matmul(self.U, self.S)

        self.L = nn.Parameter(torch.randn(output_size, rank), requires_grad = True)
        self.L.data = torch.matmul(self.V, self.S.T)

        # initialize bias
        self.bias = nn.Parameter(torch.randn(output_size))


    def forward(self, x, dlrt_step):
        """Returns the output of the layer. The formula implemented is output = U*S*V'*x + bias.
        Args:
            x: input to layer
        Returns: 
            output of layer
        """
        if dlrt_step == 'K':
            xUS = torch.matmul(x, self.K)
            out = torch.matmul(xUS, self.V.T)

        elif dlrt_step == 'L':
            xU = torch.matmul(x, self.U )
            out = torch.matmul(xU, self.L.T)

        elif dlrt_step == 'test':
            xU = torch.matmul(x, self.U )
            xUS = torch.matmul(xU, self.S)
            out = torch.matmul(xUS, self.V.T)

        #print("out.shape:", out.shape)
        return out + self.bias


    @torch.no_grad()
    def step(self, learning_rate, dlrt_step):
        """Performs a steepest descend training update on specified low-rank factors
        Args:
            learning_rate: learning rate for training
            dlrt_step: sepcifies step that is taken. Can be 'K', 'L' or 'S'
        """          
        # perform K-step
        if dlrt_step == 'K':
            self.U0.data = self.U.data
            self.K.data = self.K - learning_rate * self.K.grad
            self.U.data , _ = torch.linalg.qr(self.K, 'reduced')

            # perform S-step
            tmp = torch.matmul(self.U.T, self.U0)
            self.S.data = tmp * self.S

            self.L.data = torch.matmul(self.V, self.S.T)    
         
        # perform L-step
        elif dlrt_step == 'L':
            self.L.data = self.L - learning_rate * self.L.grad
            self.V.data , tmps = torch.linalg.qr(self.L, 'reduced')
            self.S.data = tmps.T
            self.K.data = torch.matmul(self.U, self.S)

            # update bias
            self.bias.data = self.bias - learning_rate * self.bias.grad

        else:
            print("Wrong step defined: ", dlrt_step)

    def set_all_zero(self):
        if self.S.grad is not None: self.S.grad.zero_()
        if self.U.grad is not None: self.U.grad.zero_()
        if self.V.grad is not None: self.V.grad.zero_()
        if self.K.grad is not None: self.K.grad.zero_()
        if self.L.grad is not None: self.L.grad.zero_()
        if self.bias.grad is not None: self.bias.grad.zero_()

    def get_layer_rank(self):
        return self.rank

    def write(self, file_name, use_txt=True):
        """Writes all weight matrices 
        Args:
            file_name: name of the file format in which weights are stored
        """
        # save as pth
        torch.save(self.U, file_name + "_U.pth")
        torch.save(self.S, file_name + "_S.pth")
        torch.save(self.V, file_name + "_V.pth")
        torch.save(self.bias, file_name + "_b.pth")

        if use_txt:
            with open(file_name + "_U.txt", 'w') as file:
                for row in self.U.data.T:
                    row_str = '\t'.join(map(str, row.tolist()))
                    file.write(row_str + '\n')
            with open(file_name + "_S.txt", 'w') as file:
                for row in self.S.data.T:
                    row_str = '\t'.join(map(str, row.tolist()))
                    file.write(row_str + '\n')
            with open(file_name + "_V.txt", 'w') as file:
                for row in self.V.data.T:
                    row_str = '\t'.join(map(str, row.tolist()))
                    file.write(row_str + '\n')
            with open(file_name + "_b.txt", 'w') as file:
                bias_str = '\t'.join(map(str, self.bias.data.tolist()))
                file.write(bias_str)


#################################################################################################################################################
### Define low-rank layer with Augmented Backward Corrected Projector Splitting Integrator ######################################################
#################################################################################################################################################

class PSI_Augmented_Backward_LowRankLayer(nn.Module):

    def __init__(self, input_size, output_size, rank, tol=1e-2):
        """Constructs a low-rank layer of the form U*S*V'*x + b, where 
           U, S, V represent the facorized weight W and b is the bias vector
        Args:
            input_size: input dimension of weight W
            output_size: output dimension of weight W, dimension of bias b
        """
        # construct parent class nn.Module
        super(PSI_Augmented_Backward_LowRankLayer, self).__init__()

        # set rank and trunction tolerance
        self.tol = tol

        self.rank = rank
        self.rmax = int(min(input_size, output_size) / 2) - 1  # vorher self.rmax = 2*rank
        rmax = self.rmax

        # initializes factorized weight
        self.U = nn.Parameter(torch.randn(input_size, rmax), requires_grad = True)
        self.S = nn.Parameter(torch.randn(rmax, rmax),  requires_grad = True)
        self.V = nn.Parameter(torch.randn(output_size, rmax),  requires_grad = True)
        self.U0 = nn.Parameter(torch.randn(input_size, rmax), requires_grad = False)

        # ensure that U and V are orthonormal
        self.U.data, _ = torch.linalg.qr(self.U, 'reduced')
        self.V.data, _ = torch.linalg.qr(self.V, 'reduced')

        # initializes combined weight
        self.K = nn.Parameter(torch.randn(input_size, rmax), requires_grad = True)
        self.K.data = torch.matmul(self.U, self.S)

        self.L = nn.Parameter(torch.randn(output_size, rmax), requires_grad = True)
        self.L.data = torch.matmul(self.V, self.S.T)

        # initializes placeholders for matrices with double-rank
        self.S1 = nn.Parameter(torch.randn(rmax, rmax),  requires_grad = False)

        # initialize bias
        self.bias = nn.Parameter(torch.randn(output_size))


    def forward(self, x, dlrt_step):
        """Returns the output of the layer. The formula implemented is output = U*S*V'*x + bias.
        Args:
            x: input to layer
        Returns: 
            output of layer
        """
        r = self.rank
        rx2 = 2*r

        if dlrt_step == 'K':
            xUS = torch.matmul(x, self.K[:,:r])
            out = torch.matmul(xUS, self.V[:,:r].T)

        elif dlrt_step == 'L':
            xU = torch.matmul(x, self.U[:,:rx2])
            out = torch.matmul(xU, self.L[:,:rx2].T)

        elif dlrt_step == 'test':
            xU = torch.matmul(x, self.U[:,:r])
            xUS = torch.matmul(xU, self.S[:r,:r])
            out = torch.matmul(xUS, self.V[:,:r].T)

        #print("out.shape:", out.shape)
        return out + self.bias


    @torch.no_grad()
    def step(self, learning_rate, dlrt_step):
        """Performs a steepest descend training update on specified low-rank factors
        Args:
            learning_rate: learning rate for training
            dlrt_step: sepcifies step that is taken. Can be 'K', 'L' or 'S'
        """   
        # Set current rank and extended rank 
        r = self.rank
        rx2 = 2*r      

        # perform K-step
        if dlrt_step == 'K':
            self.U0.data[:,:r] = self.U.data[:,:r]
            self.K.data[:,:r] = self.K[:,:r] - learning_rate * self.K.grad[:,:r]
            self.U.data[:,:rx2] , _ = torch.linalg.qr(torch.cat((self.K.data[:,:r], self.U0.data[:,:r]), 1), 'reduced')

            # perform S-step
            tmp = torch.matmul(self.U[:,:rx2].T, self.U0[:,:r])
            self.S.data[:rx2,:r] = torch.matmul(tmp, self.S[:r,:r])

            self.L.data[:,:rx2] = torch.matmul(self.V[:,:r], self.S[:rx2,:r].T)    
         
        # perform L-step
        elif dlrt_step == 'L':

            self.L.data[:,:rx2] = self.L[:,:rx2] - learning_rate * self.L.grad[:,:rx2]
            self.V.data[:,:rx2] , tmps = torch.linalg.qr(self.L[:,:rx2], 'reduced')
            self.S.data[:rx2, :rx2] = tmps.T

            self.truncate()

            # update bias
            self.bias.data = self.bias - learning_rate * self.bias.grad

        else:
            print("Wrong step defined: ", dlrt_step)

    
    @torch.no_grad()
    def truncate(self):
        r = self.rank
        new_rank = r
        rx2 = 2 * r

        # Truncation Step
        M, D, N_transpose = torch.linalg.svd(self.S[:rx2,:rx2])

        tol = self.tol
        if(tol > 0):
            # add values from smalles to largest until sum is larger than the tolerance
            treshold = tol * torch.linalg.norm(D)
            for j in range(0, rx2):
            # begins with full D, cuts from beginning
                if torch.linalg.norm(D[j:rx2]) < treshold:
                    new_rank = j
                    break  
        else: 
            print("tolerance of negative value")

        # check if new_rank is valid
        # 10 is chosen as the minimal rank possible
        if(new_rank < 10):
            new_rank = r
        elif(new_rank > int(self.rmax/2)):
            new_rank = int(self.rmax/2)
        
        # Update U, S, V, and rank
        self.S.data[:new_rank, :new_rank] = torch.diag(D[:new_rank])
        self.U.data[:, :new_rank] = torch.matmul(self.U[:,:rx2], M[:, :new_rank])
        self.V.data[:, :new_rank] = torch.matmul(self.V[:,:rx2], N_transpose.T[:, :new_rank])

        self.K.data[:, :new_rank] = torch.matmul(self.U[:, :new_rank], self.S[:new_rank, :new_rank])

        self.rank = new_rank

    def set_all_zero(self):
        if self.S.grad is not None: self.S.grad.zero_()
        if self.U.grad is not None: self.U.grad.zero_()
        if self.V.grad is not None: self.V.grad.zero_()
        if self.K.grad is not None: self.K.grad.zero_()
        if self.L.grad is not None: self.L.grad.zero_()
        if self.bias.grad is not None: self.bias.grad.zero_()

    def get_layer_rank(self):
        return self.rank

    def write(self, file_name, use_txt=True):
        """Writes all weight matrices 
        Args:
            file_name: name of the file format in which weights are stored
        """      
        # save as pth
        torch.save(self.U, file_name + "_U.pth")
        torch.save(self.S, file_name + "_S.pth")
        torch.save(self.V, file_name + "_V.pth")
        torch.save(self.bias, file_name + "_b.pth")

        if use_txt:
            with open(file_name + "_U.txt", 'w') as file:
                for row in self.U.data.T:
                    row_str = '\t'.join(map(str, row.tolist()))
                    file.write(row_str + '\n')
            with open(file_name + "_S.txt", 'w') as file:
                for row in self.S.data.T:
                    row_str = '\t'.join(map(str, row.tolist()))
                    file.write(row_str + '\n')
            with open(file_name + "_V.txt", 'w') as file:
                for row in self.V.data.T:
                    row_str = '\t'.join(map(str, row.tolist()))
                    file.write(row_str + '\n')
            with open(file_name + "_b.txt", 'w') as file:
                bias_str = '\t'.join(map(str, self.bias.data.tolist()))
                file.write(bias_str)
                
