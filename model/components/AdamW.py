import torch 
import math 

class AdamW:
    def __init__(self, params, lr: float = 3e-4, betas: tuple = (0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.1):
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps 
        self.weight_decay = weight_decay 
        self.t = 0 


        self.params = [p for p in params if p.requires_grad]  # fix: was `requires.grad`


        # initializing the moment buffers - m and v per parameter 

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params] 

    def zero_grad(self):
        """
        Set all parameter gradients to None (more efficient than setting to zero)"""
        for p in self.params:
            p.grad = None 

    
    @torch.no_grad() # Avoid tracking operations in AutoGrad 
    def step(self):
        """Perform one AdamW update step""" 

        self.t += 1


        for i, p in enumerate(self.params):
            if p.grad is None:
                continue  # skip the frozen parameters
            
            g = p.grad  # fix: grad is a property, not callable

            # Step 1: update the first momentum (momentum)
            # Exponential moving average of gradient
            self.m[i].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)

            # step 2: update second momentum (variance)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)  # fix: in-place addcmul_

            # step 3: bias-corrected -- at early steps, m and v are near zero. We correct this.
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)


            # step 4: AdamW update

            # Weight decay: shrink weights directly (decoupled from gradient)
            # w = w - lr * weight_decay * w
            p.mul_(1.0 - self.lr * self.weight_decay)

            # Gradient step: subtract scaled corrected momentum (gradient descent = minimize loss)
            # w = w - lr * m_hat / (sqrt(v_hat) + eps)
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)  # fix: in-place addcdiv_, negative lr
            






