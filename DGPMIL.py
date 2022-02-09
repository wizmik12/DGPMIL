import numpy as np

import torch
torch.manual_seed(10)
from torch.nn import Linear
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP

# Function for predicting in batches
def batch_means_classification(model, X):
    n_batches = max(int(X.shape[0]/1000.), 1)
    prob_list = []
    for X_batch in zip(torch.split(X, n_batches)):
        #X_batch = torch.Tensor(X_batch)
        X_batch =  torch.squeeze(X_batch[0], 0)
        prob = model(X_batch).mean.mean(0)
        prob_list.append(prob)
    prob = torch.cat(prob_list, 0)
    return prob

# Function to update the distribution of instance labels
def update_q_y(q_y, model, X, ind_bag, T):
    # Prob of positive class
    q_n = q_y[:,1]
    # Mean of the latent variable
    with torch.no_grad():
        mean_f_L = []
        print("estimating labels")
        for x_batch, y_batch in X:
            mean_f_L.append(model(x_batch).mean.mean(0))
    print("labels estimated")
    mean_f_L = torch.cat(mean_f_L, 0)
    q_n_estimated = torch.zeros(len(q_n))
    Emax = torch.zeros(len(q_n))

    if torch.cuda.is_available():
         q_n_estimated, Emax = q_n_estimated.cuda(), Emax.cuda()

    print("bag")
    for b in np.unique(ind_bag):
        mask_bag = ind_bag == b
        q_n_bag = q_n[mask_bag]
        max_bag = torch.argmax(q_n_bag)
        tmp = torch.repeat_interleave(q_n_bag[max_bag], len(q_n_bag))
        q_n_bag[max_bag] = 0
        max_bag2 = torch.argmax(q_n_bag)
        tmp[max_bag] = q_n_bag[max_bag2]
        Emax[mask_bag] = tmp
        q_n_estimated[mask_bag] = torch.sigmoid(mean_f_L[mask_bag] + np.log(100) * (2 * T[mask_bag][0] + Emax[mask_bag] -
                                                  2 * T[mask_bag][0] * Emax[mask_bag] - 1))
    return torch.vstack((1-q_n_estimated, q_n_estimated)).T

#num_output_dims = 2

# GP hidden layer class
class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(DeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

        self.linear_layer = Linear(input_dims, 1)

    def forward(self, x):
        mean_x = self.mean_module(x) # self.linear_layer(x).squeeze(-1)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)
        return super().__call__(x, are_samples=bool(len(other_inputs)))

# DGPMIL class
class DeepGPMIL(DeepGP):
    def __init__(self, x_train_shape, dims, num_inducing=128):
        # Define L hidden-layers of a L+1-layer DGP
        #dims = dims.copy()
        dims.append(None) # The last layer has None output_dims
        means = (len(dims)-1)*['linear'] + ['constant'] # The last layer with constant mean
        hidden_layers = torch.nn.ModuleList([DeepGPHiddenLayer(
            input_dims=x_train_shape,
            output_dims=dims[0],
            mean_type=means[0],
            num_inducing=num_inducing,
            )])
        for l in range(len(dims)-1):
            hidden_layers.append(DeepGPHiddenLayer(
                input_dims=hidden_layers[-1].output_dims,
                output_dims=dims[l+1],
                mean_type=means[l+1],
                num_inducing = num_inducing,
                ))

        super().__init__()

        self.hidden_layers = hidden_layers
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    def forward(self, inputs):
        output = self.hidden_layers[0](inputs)
        for hid_layer in self.hidden_layers[1:]:
            output = hid_layer(output)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)
