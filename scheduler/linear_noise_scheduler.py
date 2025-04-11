import torch


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
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0) # cummulative product of alphas
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
        
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
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
        
    def sample_prev_timestep_from_noise(self, xt, noise_pred, t):
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
        

    def sample_prev_timestep_from_x0(self, xt, x0_pred, t):
        """
        Compute x_{t-1} from the current sample xt and the predicted x0.
        
        Based on the DDPM reverse process formulation (see Ho et al., 2020):
        
        μₜ(x_t, x₀) = (√(ᾱ₍t₋₁₎) * β_t / (1 - ᾱ_t)) * x0_pred 
                        + (√(α_t) * (1 - ᾱ₍t₋₁₎) / (1 - ᾱ_t)) * xt
                        
        σₜ² = ((1 - ᾱ₍t₋₁₎) / (1 - ᾱ_t)) * β_t
        
        Args:
            xt (torch.Tensor): Current sample at timestep t, shape (B, C, H, W).
            x0_pred (torch.Tensor): Predicted x₀ from the model, same shape as xt.
            scheduler: An object that holds the diffusion hyperparameters and precomputed tensors:
                    - betas: 1D tensor of length T.
                    - alphas: 1D tensor computed as (1 - betas).
                    - alpha_cum_prod: 1D tensor representing cumulative products of alphas.
            t (int): Current timestep (0 <= t < num_timesteps).
            
        Returns:
            torch.Tensor: Sample for timestep t-1 of shape (B, C, H, W).
        """

        # For t > 0, use the previous cumulative product; for t == 0, define ᾱ₍t₋₁₎ as 1.
        if t > 0:
            alpha_cumprod_t_prev = self.alpha_cum_prod[t - 1]
        else:
            alpha_cumprod_t_prev = torch.tensor(1.0)
        sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
        
        # Compute the posterior mean μₜ (the reverse process mean) using the x₀ prediction.
        coeff1 = (sqrt_alpha_cumprod_t_prev.to(xt.device) * self.betas[t].to(xt.device)) / (1 - self.alpha_cum_prod[t].to(xt.device))
        coeff2 = (self.sqrt_alphas[t].to(xt.device) * (1 - alpha_cumprod_t_prev.to(xt.device))) / (1 - self.alpha_cum_prod[t].to(xt.device))
        mean = coeff1 * x0_pred + coeff2 * xt

        if t == 0:
            return mean
        else:
            # Compute the posterior variance and sample noise.
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)

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

