import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

# Generate synthetic data with heteroscedastic noise
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2 * np.pi)) + torch.randn(train_x.size()) * (0.1 + train_x)  # Variable noise

# Define the main GP model for predicting the mean
class MeanGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        )
        super(MeanGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Define the noise GP model to predict heteroscedastic noise levels
class NoiseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        )
        super(NoiseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Define a Multitask model to combine mean and noise GPs
class HeteroscedasticGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, mean_gp, noise_gp):
        super(HeteroscedasticGPModel, self).__init__(mean_gp.variational_strategy)
        self.mean_gp = mean_gp
        self.noise_gp = noise_gp

    def forward(self, x):
        mean_pred = self.mean_gp(x)
        noise_pred = self.noise_gp(x).exp()  # Use exp to ensure positivity of variance
        return gpytorch.distributions.MultivariateNormal(mean_pred.mean, mean_pred.lazy_covariance_matrix.add_jitter(noise_pred))

# Initialize inducing points and models
inducing_points = train_x[::10].unsqueeze(-1)  # 10 inducing points for each model
mean_gp = MeanGPModel(inducing_points)
noise_gp = NoiseGPModel(inducing_points)
model = HeteroscedasticGPModel(mean_gp, noise_gp)

# Use Variational ELBO as the marginal likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

# Use Adam optimizer
optimizer = torch.optim.Adam([
    {'params': mean_gp.parameters()},
    {'params': noise_gp.parameters()},
], lr=0.01)

# Training procedure
model.train()
likelihood.train()
training_iterations = 200
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print(f'Iter {i + 1}/{training_iterations} - Loss: {loss.item():.3f}')
    optimizer.step()

# Set models to evaluation mode
model.eval()
likelihood.eval()

# Test points for predictions
test_x = torch.linspace(0, 1, 100)

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = model(test_x)
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Observed Data')
plt.plot(test_x.numpy(), mean.numpy(), 'b', label='Mean Prediction')
plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.3, color='blue', label='Confidence Interval')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Heteroscedastic Gaussian Process Regression")
plt.show()
