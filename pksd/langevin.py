
import torch
from torch.distributions import Normal
#@title Langevin file functions
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sqrtm(A, eps=1e-12):
    """
    Compute the principal square root of a symmetric PSD matrix.

    Args:
        A: (..., n, n) symmetric positive semidefinite matrix
        eps: Eigenvalues smaller than eps are set to zero.

    Returns:
        sqrtA: (..., n, n) such that sqrtA @ sqrtA ≈ A
    """
    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(A)
    # Remove tiny negative eigenvalues due to numerical errors
    eigvals = torch.clamp(eigvals, min=eps)
    # Square root of eigenvalues
    sqrt_eigvals = torch.sqrt(eigvals)
    # Reconstruct matrix square root
    sqrtA = eigvecs @ torch.diag_embed(sqrt_eigvals) @ eigvecs.transpose(-1, -2)

    return sqrtA

def prepare_proposal_input(mode1: torch.Tensor,
                           mode2: torch.Tensor,
                           hess1_inv: torch.Tensor,
                           hess2_inv: torch.Tensor):
    """
    Given two modes and the local inverse Hessians, compute the
    quantities needed to construct the proposals.

    Args:
        mode1, mode2: (dim,)
        hess1_inv, hess2_inv: (dim, dim), inverse Hessians at the modes
    """

    device = hess1_inv.device
    dtype = hess1_inv.dtype

    # If Hessian estimates were not stable, set to I_d
    if torch.linalg.det(hess1_inv) <= 0:
        hess1_inv = torch.eye(hess1_inv.shape[0], device=device, dtype=dtype)

    if torch.linalg.det(hess2_inv) <= 0:
        hess2_inv = torch.eye(hess2_inv.shape[0], device=device, dtype=dtype)

    # Matrix square roots
    hess1_inv_sqrt = sqrtm(hess1_inv)
    hess2_inv_sqrt = sqrtm(hess2_inv)

    # Inverse square roots
    hess1_sqrt = torch.linalg.inv(hess1_inv_sqrt)
    hess2_sqrt = torch.linalg.inv(hess2_inv_sqrt)

    # Determinants
    hess1_sqrt_det = torch.linalg.det(hess1_sqrt)
    hess1_inv_sqrt_det = torch.linalg.det(hess1_inv_sqrt)

    hess2_sqrt_det = torch.linalg.det(hess2_sqrt)
    hess2_inv_sqrt_det = torch.linalg.det(hess2_inv_sqrt)

    hess_dict = {
        "mode1": mode1,
        "hess1_sqrt": hess1_sqrt,
        "hess1_inv_sqrt": hess1_inv_sqrt,
        "hess1_sqrt_det": hess1_sqrt_det,
        "hess1_inv_sqrt_det": hess1_inv_sqrt_det,

        "mode2": mode2,
        "hess2_sqrt": hess2_sqrt,
        "hess2_inv_sqrt": hess2_inv_sqrt,
        "hess2_sqrt_det": hess2_sqrt_det,
        "hess2_inv_sqrt_det": hess2_inv_sqrt_det,
    }

    return hess_dict


def prepare_proposal_input_all(mode_list: list, inv_hess_list: list):
    """
    Given a list of modes and the local inverse Hessians, compute the
    quantities needed to construct the proposals.

    Args:
        mode_list: list of modes of shape (dim,)
        inv_hess_list: list of inverse Hessians at the modes,
                       each of shape (dim, dim)
    """

    # Stack modes: (nmodes, dim)
    modes = torch.stack(mode_list)

    device = inv_hess_list[0].device
    dtype = inv_hess_list[0].dtype

    # If Hessian estimates were not stable, set to I_d
    for i, inv_hess in enumerate(inv_hess_list):
        if torch.linalg.det(inv_hess) <= 0:
            inv_hess_list[i] = torch.eye(
                inv_hess.shape[0],
                device=device,
                dtype=dtype,
            )

    # Matrix square roots of inverse Hessians
    inv_hess_sqrt_list = [sqrtm(x) for x in inv_hess_list]
    inv_hess_sqrt = torch.stack(inv_hess_sqrt_list)   # (nmodes, dim, dim)

    # Square roots of Hessians = inverse of sqrt(inv_hessian)
    hess_sqrt_list = [torch.linalg.inv(x) for x in inv_hess_sqrt_list]
    hess_sqrt = torch.stack(hess_sqrt_list)           # (nmodes, dim, dim)

    # Determinants
    hess_sqrt_det_list = [torch.linalg.det(x) for x in hess_sqrt_list]
    hess_sqrt_det = torch.stack(hess_sqrt_det_list)   # (nmodes,)

    inv_hess_sqrt_det_list = [torch.linalg.det(x) for x in inv_hess_sqrt_list]
    inv_hess_sqrt_det = torch.stack(inv_hess_sqrt_det_list)   # (nmodes,)

    modes_dict = {
        "modes": modes,
        "hess_sqrt": hess_sqrt,
        "inv_hess_sqrt": inv_hess_sqrt,
        "hess_sqrt_det": hess_sqrt_det,
        "inv_hess_sqrt_det": inv_hess_sqrt_det,
    }

    return modes_dict


class MCMC:
    def __init__(self, log_prob: callable) -> None:
        self.log_prob = log_prob
        self.x = None
        self.noise_dist = Normal(0.0, 1.0)

    def log_transition_kernel(self, xp: torch.Tensor, x: torch.Tensor):
        """
        Compute log k(x'|x), where k is the transition kernel.

        Args:
            xp: Proposed samples (n x dim)
            x: Current samples (n x dim)

        Returns:
            log_kernel: n
        """
        raise NotImplementedError(
            f"Sampler class '{self.__class__.__name__}' does not provide a "
            f"'transition kernel'"
        )

    def compute_accept_prob(self,x_proposed: torch.Tensor,x_current: torch.Tensor,log_det_jacobian: torch.Tensor,**kwargs,):
        """
        Compute the Metropolis-Hastings acceptance probability.

        log(min(1,
            k(x|x') p(x')
            ----------------
            k(x'|x) p(x)
        ))
        """

        log_numerator = (self.log_prob(x_proposed) + self.log_transition_kernel(xp=x_current,x=x_proposed,**kwargs,)+ log_det_jacobian)
        log_denominator = (self.log_prob(x_current) + self.log_transition_kernel( xp=x_proposed, x=x_current,**kwargs,))

        log_prob = torch.minimum(torch.zeros_like(log_numerator),log_numerator - log_denominator,)

        assert log_prob.shape == (x_proposed.shape[0],x_proposed.shape[1],)
        return torch.exp(log_prob)

    def transit(self, x_proposed: torch.Tensor,x_current: torch.Tensor,accept_prob: torch.Tensor,):
        """
        Perform the Metropolis accept/reject step.

        Args:
            x_proposed: (njumps, n, dim)
            x_current: (njumps, n, dim)
            accept_prob: (njumps, n)

        Returns:
            x_next: (njumps, n, dim)
            if_accept: (njumps, n)
        """

        unif = torch.rand(accept_prob.shape,device=accept_prob.device, dtype=accept_prob.dtype,)
        cond = (unif < accept_prob).unsqueeze(-1)
        x_next = torch.where(cond, x_proposed, x_current)
        if_accept = cond.squeeze(-1).float()
        assert x_next.shape == x_proposed.shape
        return x_next, if_accept

class RandomWalkMH(MCMC):
  def __init__(self, log_prob: callable) -> None:
      super().__init__(log_prob)

  def run( self,steps: int,x_init: torch.Tensor,std,verbose: bool = False,**kwargs,):
      n, dim = x_init.shape
      device = x_init.device
      dtype = x_init.dtype
      # theta: (njumps,)
      if not isinstance(std, torch.Tensor):
          theta = torch.tensor(std, device=device, dtype=dtype)
      else:
          theta = std.to(device=device, dtype=dtype)

      theta = theta.reshape(-1)
      # (njumps, n, dim)
      x_init = x_init.unsqueeze(0).repeat(theta.shape[0], 1, 1)

      self.x = [x_init]
      self.accept_prob = []
      self.if_accept = []

      if "ind_pair_list" in kwargs:
          npairs = len(kwargs["ind_pair_list"])
          ind_prob = torch.ones(npairs, device=device) / npairs

          cat = Categorical(probs=ind_prob)
          self.ind_pair_sample = cat.sample((steps - 1, n))   # (steps-1, n)

          self.ind_pairs = torch.tensor(kwargs["ind_pair_list"],device=device, dtype=torch.long,)

      iterator = trange(steps - 1) if verbose else range(steps - 1)
      for t in iterator:

          self.t = t

          # (njumps, n, dim)
          x_current = self.x[t]

          xp_next, log_det_jacobian = self.proposal(x_current=x_current, theta=theta, **kwargs,)
          accept_prob = self.compute_accept_prob(x_proposed=xp_next,x_current=x_current,log_det_jacobian=log_det_jacobian, **kwargs, )
          self.accept_prob.append(accept_prob)

          x_next, if_accept = self.transit(x_proposed=xp_next,x_current=x_current,accept_prob=accept_prob,)
          self.if_accept.append(if_accept)
          self.x.append(x_next)

      self.x = torch.stack(self.x, dim=1)
      self.accept_prob = torch.stack(self.accept_prob, dim=1)
      self.if_accept = torch.stack(self.if_accept, dim=1)

      self.accept_prob = self.accept_prob.squeeze()
      self.if_accept = self.if_accept.squeeze()

  def log_transition_kernel(  self, xp: torch.Tensor,x: torch.Tensor, **kwargs,):
      """
      Compute log k(x'|x).
      """
      log_prob = torch.log(torch.tensor(0.5,device=xp.device,dtype=xp.dtype,)) * torch.ones(xp.shape[-2],device=xp.device, dtype=xp.dtype,)
      return log_prob

  def proposal(self, x_current, theta, **kwargs):
      """
      Default proposal.
      x_current: (njumps, n, dim)
      """
      # (njumps,1,1)
      theta = theta.unsqueeze(-1).unsqueeze(-1)
      if "ind_pair_list" not in kwargs:
        raise ValueError("ind_pair_list not found in kwargs.")

      # (n,)
      ind_pair_ind = self.ind_pair_sample[self.t]

      # (n,2)
      ind_pair = self.ind_pairs[ind_pair_ind]
      # Gather quantities for each pair
      mode1 = kwargs["modes"][ind_pair[:, 0]]
      mode2 = kwargs["modes"][ind_pair[:, 1]]

      inv_root_cov_1 = kwargs["hess_sqrt"][ind_pair[:, 0]]
      root_cov_2 = kwargs["inv_hess_sqrt"][ind_pair[:, 1]]

      inv_root_cov_1_det = kwargs["hess_sqrt_det"][ind_pair[:, 0]]
      root_cov_2_det = kwargs["inv_hess_sqrt_det"][ind_pair[:, 1]]

      # Proposal
      x_current_c = x_current - theta * mode1

      # (njumps,n,1,dim)
      x_current_c = x_current_c.unsqueeze(-2)
      x_current_c_scaled = (x_current_c@ inv_root_cov_1 @ root_cov_2)

      assert x_current_c_scaled.shape == (theta.shape[0],x_current.shape[-2],1,x_current.shape[-1],)

      xp_next = x_current_c_scaled.squeeze(-2) + theta * mode2

      det_jacobian = inv_root_cov_1_det * root_cov_2_det
      log_det_jacobian = torch.log(det_jacobian)
      return xp_next, log_det_jacobian

class RandomWalkBarker(RandomWalkMH):
    def __init__(self, log_prob: callable) -> None:
        super().__init__(log_prob)

    def compute_accept_prob(self,x_proposed: torch.Tensor,x_current: torch.Tensor,log_det_jacobian: torch.Tensor,**kwargs,):
        """
        Compute the Barker acceptance probability:
                              k(x|x') p(x')
        α(x,x') =        -----------------------
                        k(x'|x) p(x) + k(x|x') p(x')
        """

        term_xp = torch.exp(self.log_prob(x_proposed)+ self.log_transition_kernel(xp=x_current,x=x_proposed,**kwargs,)+ log_det_jacobian)  # (njumps, n)
        term_x = torch.exp(self.log_prob(x_current)+ self.log_transition_kernel(xp=x_proposed,x=x_current,**kwargs,))  # (njumps, n)

        prob = term_xp / (term_xp + term_x)
        assert prob.shape == (x_proposed.shape[0],x_proposed.shape[1],)
        return prob