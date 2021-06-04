#
# Robust pooling
#
# y(x) = argmin_u f(x, u)
#
# where f(x, u) = sum_{i=1}^n phi(u - x_i; scale)
# with penalty function phi in
# {quadratic, pseudo-huber, huber, welsch, truncated quadratic}
#
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Stephen Gould <stephen.gould@anu.edu.au>
#

import torch
from ddn.pytorch.robust_loss_pytorch.adaptive import *
from ddn.pytorch.robust_loss_pytorch.general import lossfun
from ddn.pytorch.node import *

class Quadratic():
    is_convex = True

    @staticmethod
    def phi(z, scale = 1.0):
        """ Quadratic penalty function

        phi(z; scale) = 0.5 * z^2

        Arguments:
            z: (b, ...) Torch tensor,
                batch of residuals

            scale: float, optional, default: 1.0,
                ignored

        Return Values:
            phi_z: (b, ...) Torch tensor,
                Quadratic penalty associated with each residual

        Complexity:
            O(1)
        """
        phi_at_z = 0.5 * torch.pow(z, 2)
        return phi_at_z

    @staticmethod
    def Dy(z, scale = 1.0):
        # Derivative of y(x) for the quadratic penalty function
        Dy_at_x = torch.ones_like(z) / (z.size(-1) * z.size(-2))
        return Dy_at_x

class PseudoHuber():
    is_convex = True

    @staticmethod
    def phi(z, scale = 1.0):
        """ Pseudo-Huber penalty function

        phi(z; scale) = scale^2 (sqrt{1 + (z / scale)^2} - 1)

        Arguments:
            z: (b, ...) Torch tensor,
                batch of residuals

            scale: float, optional, default: 1.0,
                ~slope of the linear region
                ~maximum residual in the quadratic region

        Return Values:
            phi_z: (b, ...) Torch tensor,
                Pseudo-Huber penalty associated with each residual

        Complexity:
            O(1)
        """
        #assert scale > 0.0, "scale must be strictly positive (%f <= 0)" % scale
        scale2 = scale * scale
        phi_at_z = scale2 * (torch.sqrt(1.0 + torch.pow(z, 2) / scale2) - 1.0)
        return phi_at_z

    @staticmethod
    def Dy(z, scale = 1.0):
        # Derivative of y(x) for the pseudo-Huber penalty function
        w = torch.pow(1.0 + torch.pow(z, 2) / (scale * scale), -1.5)
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        Dy_at_x = torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum.expand_as(w)))
        Dy_at_scale=None
        if scale.requires_grad:
            w_scale=torch.pow(z,3)/(torch.pow(torch.pow(z/scale,2)+1,1.5)*torch.pow(scale,3))
            w_scale=-torch.sum(w_scale,dim=1,keepdim=True)
            Dy_at_scale = torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w_scale), w_scale.div(w_sum.expand_as(w_scale)))
        return Dy_at_x,Dy_at_scale
class Huber():
    is_convex = True

    @staticmethod
    def phi(z, scale = 1.0):
        """ Huber penalty function

                        / 0.5 z^2 for |z| <= scale
        phi(z; scale) = |
                        \ scale (|z| - 0.5 scale) else

        Arguments:
            z: (b, ...) Torch tensor,
                batch of residuals

            scale: float, optional, default: 1.0,
                slope of the linear region
                maximum residual in the quadratic region

        Return Values:
            phi_z: (b, ...) Torch tensor,
                Huber penalty associated with each residual

        Complexity:
            O(1)
        """
        assert scale > 0.0, "scale must be strictly positive (%f <= 0)" % scale
        z = z.abs()
        phi_at_z = torch.where(z <= scale, 0.5 * torch.pow(z, 2), scale * (z - 0.5 * scale))
        return phi_at_z

    # @staticmethod
    def Dy(z, scale = 1.0):
        # Derivative of y(x) for the Huber penalty function
        w = torch.where(z.abs() <= scale, torch.ones_like(z), torch.zeros_like(z))
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        Dy_at_x = torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum.expand_as(w)))
        Dy_at_scale=None
        if scale.requires_grad:
            w_scale= -torch.where(z.abs()<=scale,torch.zeros_like(z),torch.where(z>=scale,torch.ones_like(z),-torch.ones_like(z)))
            w_scale=torch.sum(w_scale,dim=1,keepdim=True)
            Dy_at_scale= torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w_scale), w_scale.div(w_sum))
        return Dy_at_x,Dy_at_scale


class Welsch():
    is_convex = False

    @staticmethod
    def phi(z, scale = 1.0):
        """ Welsch penalty function

        phi(z; scale) = 1 - exp(-0.5 * z^2 / scale^2)

        Arguments:
            z: (b, ...) Torch tensor,
                batch of residuals

            scale: float, optional, default: 1.0,
                ~maximum residual in the quadratic region

        Return Values:
            phi_z: (b, ...) Torch tensor,
                Welsch penalty associated with each residual

        Complexity:
            O(1)
        """
        assert scale > 0.0, "scale must be strictly positive (%f <= 0)" % scale
        phi_at_z = 1.0 - torch.exp(-torch.pow(z, 2) / (2.0 * scale * scale))
        return phi_at_z

    @staticmethod
    def Dy(z, scale = 1.0):
        # Derivative of y(x) for the Welsch penalty function
        scale2 = scale * scale
        z2_on_scale2 = torch.pow(z, 2) / scale2
        w = (1.0 - z2_on_scale2) * torch.exp(-0.5 * z2_on_scale2) / scale2
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        Dy_at_x = torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum.expand_as(w)))
        Dy_at_x = torch.clamp(Dy_at_x, -1.0, 1.0) # Clip gradients to +/- 1
        Dy_at_scale=None
        if scale.requires_grad:
            w_scale = (torch.exp(-0.5 * z2_on_scale2)*(2*z*scale2-torch.pow(z,3))/(scale**5))
            w_scale=torch.sum(w_scale,dim=1,keepdim=True)
            Dy_at_scale =torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w_scale), w_scale.div(w_sum))
            Dy_at_scale = torch.clamp(Dy_at_scale, -1.0, 1.0)
        return Dy_at_x,Dy_at_scale

class TruncatedQuadratic():
    is_convex = False

    @staticmethod
    def phi(z, scale = 1.0):
        """ Truncated quadratic penalty function

                        / 0.5 z^2 for |z| <= scale
        phi(z; scale) = |
                        \ 0.5 scale^2 else

        Arguments:
            z: (b, ...) Torch tensor,
                batch of residuals

            scale: float, optional, default: 1.0,
                maximum residual in the quadratic region

        Return Values:
            phi_z: (b, ...) Torch tensor,
                Truncated quadratic penalty associated with each residual

        Complexity:
            O(1)
        """
        assert scale > 0.0, "scale must be strictly positive (%f <= 0)" % scale
        z = z.abs()
        phi_at_z = torch.where(z <= scale, 0.5 * torch.pow(z, 2), 0.5 * scale * scale * torch.ones_like(z))
        return phi_at_z

    @staticmethod
    def Dy(z, scale = 1.0):
        # Derivative of y(x) for the truncated quadratic penalty function
        w = torch.where(z.abs() <= scale, torch.ones_like(z), torch.zeros_like(z))
        w_sum = w.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).expand_as(w)
        Dy_at_x = torch.where(w_sum.abs() <= 1e-9, torch.zeros_like(w), w.div(w_sum))
        return Dy_at_x

class AdaptiveAndGeneral():
    is_convex = False

    @staticmethod
    def phi(z, alpha = 1.0, scale = 1.0):
        z=z.squeeze(-1)
        phi =  lossfun(z, alpha,scale, approximate=False)
        return phi

    @staticmethod
    def Dy(z, alpha = 1.0, scale = 1.0):
        abs_a_minus=torch.abs(alpha-2)
        scale_pow=torch.pow(scale,2)
        z_pow=torch.pow(z,2)
        term_2=(z_pow/(abs_a_minus*scale_pow))+1
        term_1=torch.pow(term_2,alpha/2)
        dyy= (abs_a_minus*((alpha-1)*z_pow+ abs_a_minus*scale_pow)*term_1)/torch.pow(z_pow+abs_a_minus*scale_pow,2)
        dyy_sum=dyy.sum(-1)
        dyy_sum=dyy_sum.unsqueeze(-1)
        dycl=None
        dyal=None
        if scale.requires_grad: 
            dyc= ((abs_a_minus*z*term_1*(2*abs_a_minus*scale_pow+alpha*z_pow))/(scale*torch.pow((abs_a_minus*scale_pow+z_pow),2)))
            dyc=dyc.sum(-1)
            dycl=dyc.unsqueeze(-1)
            dycl=dycl/dyy_sum
            dycl=torch.clamp(dycl,-1.0,1.0)
        if alpha.requires_grad:
            dya=z*torch.pow(term_2,alpha/2-1)*(torch.log(term_2)/2-((z_pow*(alpha/2-1))/(scale_pow*term_2*abs_a_minus*(alpha-2))))
            dya=(dya/scale_pow).sum(-1)
            dyal=dya.unsqueeze(-1)
            dyal=-dyal/dyy_sum
            dyal=torch.clamp(dyal,-1.0,1.0)
        return dyy/dyy_sum, dyal,dycl

class RobustGlobalPool2dFn(torch.autograd.Function):
    """
    A function to globally pool a 2D response matrix using a robust penalty function
    """
    @staticmethod
    def runOptimisation(x, y, method, scale_scalar):
        with torch.enable_grad():
            opt = torch.optim.LBFGS([y],
                                    lr=1, # Default: 1
                                    max_iter=100, # Default: 20
                                    max_eval=None, # Default: None
                                    tolerance_grad=1e-05, # Default: 1e-05
                                    tolerance_change=1e-09, # Default: 1e-09
                                    history_size=100, # Default: 100
                                    line_search_fn=None # Default: None, Alternative: "strong_wolfe"
                                    )
            def reevaluate():
                opt.zero_grad()
                # Sum cost function across residuals and batch (all fi are positive)
                f = method.phi(y.unsqueeze(-1).unsqueeze(-1) - x, scale=scale_scalar).sum()
                f.backward()
                return f
            opt.step(reevaluate)
        return y

    @staticmethod
    def forward(ctx, x, method, scale):
        input_size = x.size()
        assert len(input_size) >= 2, "input must at least 2D (%d < 2)" % len(input_size)
        scale_scalar = scale.detach()
        #assert scale.item() > 0.0, "scale must be strictly positive (%f <= 0)" % scale.item()
        x = x.detach()
        x = x.flatten(end_dim=-3) if len(input_size) > 2 else x
        # Handle non-convex functions separately
        if method.is_convex:
            # Use mean as initial guess
            y = x.mean([-2, -1]).clone().requires_grad_()
            y = RobustGlobalPool2dFn.runOptimisation(x, y, method, scale_scalar)
        else:
            # Use mean and median as initial guesses and choose the best
            # ToDo: multiple random starts
            y_mean = x.mean([-2, -1]).clone().requires_grad_()
            y_mean = RobustGlobalPool2dFn.runOptimisation(x, y_mean, method, scale_scalar)
            y_median = x.flatten(start_dim=-2).median(dim=-1)[0].clone().requires_grad_()
            y_median = RobustGlobalPool2dFn.runOptimisation(x, y_median, method, scale_scalar)
            f_mean = method.phi(y_mean.unsqueeze(-1).unsqueeze(-1) - x, scale=scale_scalar).sum(-1).sum(-1)
            f_median = method.phi(y_median.unsqueeze(-1).unsqueeze(-1) - x, scale=scale_scalar).sum(-1).sum(-1)
            y = torch.where(f_mean <= f_median, y_mean, y_median)
        y = y.detach()
        z = (y.unsqueeze(-1).unsqueeze(-1) - x).clone()
        ctx.method = method
        ctx.input_size = input_size
        ctx.save_for_backward(z, scale)
        return y.reshape(input_size[:-2]).clone()

    @staticmethod
    def backward(ctx, grad_output):
        z, scale = ctx.saved_tensors
        input_size = ctx.input_size
        method = ctx.method
        grad_input_x = None
        if ctx.needs_input_grad[0]:
            # Flatten:
            grad_output = grad_output.detach().flatten(end_dim=-1)
            grad_output= grad_output.unsqueeze(-1).unsqueeze(-1)
            # Use implicit differentiation to compute derivative:
            dx,dscale = method.Dy(z, scale)
            grad_input_x = dx * grad_output
            if dscale!=None:
                grad_input_scale =dscale *grad_output
                grad_input_scale =grad_input_scale.sum(0).squeeze(-1)
            else:
                grad_input_scale=None
            grad_input_x = grad_input_x.reshape(input_size)
        return grad_input_x, None,grad_input_scale

class RobustGlobalPool2d(torch.nn.Module):
    def __init__(self, method, scale=1.0,is_image=True):
        super(RobustGlobalPool2d, self).__init__()
        self.method = method
        self.register_buffer('scale', torch.tensor([scale]))
        self.is_image=is_image

    def forward(self, input):
        #print('input shape:',input.shape)
        #print(aaaa)
        if self.is_image:
            #print('aa')
            input=input.flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)
            # print('value',input.max(dim=2)[0].mean(),input.min(dim=2)[0].mean())
            # print('variance',input.var(dim=2)[0].mean())
        output= RobustGlobalPool2dFn.apply(input,
                                          self.method,
                                          self.scale
                                          )
        if self.is_image:
            output=output.unsqueeze(-1).unsqueeze(-1)
        #print('output shape',output.shape)
        #print(asdasd)
        return output

    def extra_repr(self):
        return 'method={}, scale={}'.format(
            self.method, self.scale
        )

class AdaptiveAndGeneralFn(torch.autograd.Function):
    """
    A function to globally pool a 2D response matrix using a robust penalty function
    """
    @staticmethod
    def runOptimisation( x,y,alpha,scale):
        #latent_alpha=latent_alpha.clone()
        #latent_scale=latent_scale.clone()
        with torch.enable_grad():
            opt = torch.optim.LBFGS([y],
                                    lr=1, # Default: 1
                                    max_iter=100, # Default: 20
                                    max_eval=None, # Default: None
                                    tolerance_grad=1e-05, # Default: 1e-05
                                    tolerance_change=1e-09, # Default: 1e-09
                                    history_size=100, # Default: 100
                                    line_search_fn=None # Default: None, Alternative: "strong_wolfe"
                                    )
            def reevaluate():
                opt.zero_grad()
                f = AdaptiveAndGeneral.phi(y.unsqueeze(-1).unsqueeze(-1) - x,alpha,scale).sum() # sum over batch elements
                f.backward()
                return f
            opt.step(reevaluate)
        return y.clone()
    

    @staticmethod
    def forward(ctx, x,alpha,scale):
        input_size = x.size()
        assert len(input_size) >= 2, "input must at least 2D (%d < 2)" % len(input_size)
        alpha_scalar=alpha.detach()
        scale_scalar=scale.detach()
        x = x.detach()
        x = x.flatten(end_dim=-3) if len(input_size) > 2 else x
        y_mean = x.mean([-2, -1]).clone().requires_grad_()
        y_mean = AdaptiveAndGeneralFn.runOptimisation(x, y_mean, alpha_scalar,scale_scalar)
        y_median = x.flatten(start_dim=-2).median(dim=-1)[0].clone().requires_grad_()
        y_median = AdaptiveAndGeneralFn.runOptimisation(x, y_median, alpha_scalar,scale_scalar)
        f_mean = AdaptiveAndGeneral.phi(y_mean.unsqueeze(-1).unsqueeze(-1) - x, alpha=alpha_scalar,scale=scale_scalar).sum(-1).sum(-1)
        f_median = AdaptiveAndGeneral.phi(y_median.unsqueeze(-1).unsqueeze(-1) - x, alpha=alpha_scalar,scale=scale_scalar).sum(-1).sum(-1)
        y = torch.where(f_mean <= f_median, y_mean, y_median)
        y = y.detach()
        z = (y.unsqueeze(-1).unsqueeze(-1) - x).clone().squeeze(-1)
        ctx.input_size = input_size
        ctx.save_for_backward(z, alpha,scale)
        return y.reshape(input_size[:-2]).clone()

    @staticmethod
    def backward(ctx, grad_output):
        z, alpha,scale = ctx.saved_tensors
        input_size = ctx.input_size
        grad_input_x = None
        grad_input_alpha = None
        grad_input_scale = None
        if ctx.needs_input_grad[0]:
            # Flatten:
            grad_output = grad_output.detach().flatten(end_dim=-1)
            grad_output=grad_output.unsqueeze(-1).unsqueeze(-1)
            dx,dalpha,dscale=AdaptiveAndGeneral.Dy(z, alpha,scale)
            dx=dx.unsqueeze(-1)
            grad_input_x = dx * grad_output
            grad_input_alpha =dalpha.unsqueeze(-1) *grad_output if dalpha!=None else None
            grad_input_scale=dscale.unsqueeze(-1)*grad_output if dscale!=None else None
            grad_input_x = grad_input_x.reshape(input_size)
            # if grad_input_alpha!=None and grad_input_alpha.shape!=alpha.shape:
            #     grad_input_alpha=grad_input_alpha.sum(0).squeeze(-1)
            # if grad_input_scale!=None and grad_input_scale.shape!=scale.shape:
            #     grad_input_scale=grad_input_scale.sum(0).squeeze(-1)
        return grad_input_x , grad_input_alpha,grad_input_scale
    def gradient(self,x,scale,y):
        x.requires_grad_()
        y.requires_grad_()
        scale.requires_grad_()
        return super().gradient(x,scale,y=y)

""" Check gradients
from torch.autograd import gradcheck

scale = 1.0
# scale = 0.2
# scale = 5.0

method = Quadratic
# method = PseudoHuber
# method = Huber
# method = Welsch # Can fail gradcheck due to numerically-necessary gradient clipping
# method = TruncatedQuadratic

robustPool = RobustGlobalPool2dFn.apply
scale_tensor = torch.tensor([scale], dtype=torch.double, requires_grad=False)
input = (torch.randn(2, 3, 7, 7, dtype=torch.double, requires_grad=True), method, scale_tensor)
test = gradcheck(robustPool, input, eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=True)
print("{}: {}".format(method.__name__, test))
"""
