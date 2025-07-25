import torch, os, sys


class LinearNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used in DDPM.
    """
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0) # cummulative product of alphas
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

        # print('betas', self.betas)
        # print('alphas', self.alphas)
        # print('alpha_cum_prod', self.alpha_cum_prod)
        # print('sqrt_alpha_cum_prod', self.sqrt_alpha_cum_prod)
        # print('sqrt_one_minus_alpha_cum_prod', self.sqrt_one_minus_alpha_cum_prod)
        # sys.exit()
        
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        
        # if batch_size > 1:
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        # else:
        #     sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t]
        #     sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t]

        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
    

    def add_noise_partial(self, original, noise, t, n_cond=2):
        r"""
        Add noise to the image but not the conditions
        :param original: Image on which noise is to be applied with conditions
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :param n_cond: number of conditions
        :return:
        """
        x_pred, x_cond = original[:, :n_cond], original[:, n_cond:]
        noisy_pred = self.add_noise(x_pred, noise[:, :n_cond], t)

        return torch.cat([noisy_pred, x_cond], dim=1).to(original.device)
        
    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the noise predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)
        
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])
        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0
        
    def sample_prev_timestep_partial(self, xt, noise_pred, t, n_cond=2):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the noise predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :param n_cond: number of conditions
        :return:
        """
        x_pred, x_cond = xt[:, :n_cond], xt[:, n_cond:]
        # xt1, _ = self.sample_prev_timestep(x_pred, noise_pred[:, :n_cond], t.squeeze(0))
        xt1, _ = self.sample_prev_timestep(x_pred, noise_pred, t.squeeze(0))

        return torch.cat([xt1, x_cond], dim=1).to(xt.device)
        
    def sample_prev_timestep_image_from_x0(self, xt, x0, t):
        r"""
            Use the x0 prediction to get
            xt-1 using xt and the x0 predicted
        :param xt: current timestep sample
        :param x0: model x0 prediction
        :param t: current timestep we are at
        :return:
        """
        mean = (self.sqrt_alpha_cum_prod.to(xt.device)[t] * x0 +
                self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * xt)
        
        if t == 0:
            return mean
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z
        
    def sample_x0_from_noise(self, xt, noise_pred, t):
        # xt and noise_pred are [B, C, H, W]
        # t is [B]
        
        # Reshape the 1D timestep-dependent tensors for broadcasting:
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t].view(-1, 1, 1, 1)
        alpha_cum = self.alpha_cum_prod.to(xt.device)[t].view(-1, 1, 1, 1)

        # Calculate x0 according to the equation:
        x0 = (xt - sqrt_one_minus_alpha * noise_pred) / torch.sqrt(alpha_cum)
        x0 = torch.clamp(x0, -1., 1.)
        
        return x0

