pi = 3.141592653
e = 2.71828
import torch
# example: Rastrigin function
class Solver():
	"""docstring for Solver"""
	def __init__(self, params):
		super(Solver, self).__init__()
		self.D = params.dim
		self.xopt = torch.torch.FloatTensor(params.xopt)
		self.s = params.s
		self.R25 = params.R25
		self.R50 = params.R50
		self.R100 = params.R100
		self.w = params.w
		self.m = params.m
		self.P = params.P
		self.func_num = params.func_num
		self.func_dict = self.init_dict()
		self.norm = self.init_norm(self.func_num)
		self.ub = params.ub
		self.lb = params.lb

	def init_dict(self):
		return {1: self.f1, 
				2: self.f2,
				3: self.f3, 
				4: self.f4,
				5: self.f5, 
				6: self.f6}

	def init_norm(self, func):
		norm_dict = { 
				1: 1e5,
				2: 50, 
				3: 1e8, 
				4: 1e6,
				5: 1e10, 
				6: 1e8
				}
		return norm_dict[func]

	def run(self, x, grad=True, is_norm=True):
		x = x.clone().detach().requires_grad_(True)

		FoM = self.forward(x, is_norm)
		if grad:
			Grad = self.backward(x)
			return (FoM.detach(), Grad.detach())
		else:
			return FoM.detach()


	def forward(self, x, is_norm=True):

		return self.func_dict[self.func_num](x, is_norm)

	def backward(self, x):
		Grad = torch.autograd.functional.jacobian(self.forward, x)

		Grad = torch.diagonal(Grad.squeeze()).t()

		return Grad


	def T_osz(self, x):
		xx = torch.log(torch.abs(x) + 1e-8)
		c1 = torch.ones_like(x)*10.0
		c2 = torch.ones_like(x)*7.9
		c1[x < 0] = 5.5
		c2[x < 0] = 3.1
		y = torch.sign(x) * torch.exp(xx + 0.049*(torch.sin(c1*xx) + torch.sin(c2*xx)))
		return y


	def T_asy(self, x, beta):
		D = x.size(1)
		idx = torch.linspace(0, 1, D).view(1, -1)

		y = torch.pow(torch.abs(x), 1+beta*idx*torch.sqrt(torch.abs(x)+1e-8))
		mask = x < 0
		y = y * (~mask) + x * mask

		return y


	def T_diag(self, alpha, D):
		idx = torch.linspace(0, 1, D).view(1, -1)
		q = torch.pow(alpha, 0.5*idx).detach()
		return q

	def Sphere_func(self, x):
		return torch.sum(x * x, dim=1, keepdim=True)

	def Elliptic_func(self, x):
		D = x.size(1)
		condition = 1e6
		coeffients = condition ** torch.linspace(0, 1, D).view(1, -1)
		x = self.T_osz(x)
		f = torch.sum(coeffients * x * x, dim=1, keepdim=True)
		return f 

	def Rastrigin_func(self, x):
		x = self.T_diag(10.0, x.size(1))*self.T_asy(self.T_osz(x), 0.2)
		f = torch.sum(x * x - 10. * torch.cos(2*pi*x) + 10., dim=1, keepdim=True)
		return f


	def Ackley_func(self, x):
		D = x.size(1)
		f = -20 * torch.exp(-0.2*torch.sqrt(torch.sum(x*x, dim=1, keepdim=True)/D)) \
			-torch.exp(torch.sum(torch.cos(2*pi*x), dim=1, keepdim=True)/D) + 20 + e
		return f

	def Schwefel_func(self, x):
		D = x.size(1)
		x = self.T_asy(self.T_osz(x), 0.2)
		mask = torch.tril(torch.ones(D, D)).unsqueeze(0)
		x = x.unsqueeze(1)

		x_sum = torch.sum(mask * x, dim=2)
		f = torch.sum(x_sum*x_sum, dim=1, keepdim=True)
		return f

	def Rosenbrock_func(self, x):
		x1 = x[:,1:]
		x2 = (x*x)[:,:-1]
		x3 = x[:,:-1]
		return torch.sum(100.0*(x1 - x2) * (x1 - x2) + (x3 - 1) * (x3 - 1), dim=1, keepdim = True) 




	def f1(self, x, is_norm=True):
		x_u = 5
		x_l = -5

		x = x * (x_u - x_l)/2.0 + (x_u + x_l)/2.0
		x = x - self.xopt

		f = self.Rastrigin_func(x)
		if is_norm:
			return f/self.norm
		else:
			return f


	def f2(self, x, is_norm=True):
		x_u = 32
		x_l = -32

		x = x * (x_u - x_l)/2.0 + (x_u + x_l)/2.0
		x = x - self.xopt

		f = self.Ackley_func(x)

		if is_norm:
			return f/self.norm
		else:
			return f



	
	
	def f3(self, x, is_norm=True):
		x_u = 5
		x_l = -5

		x = x * (x_u - x_l)/2.0 + (x_u + x_l)/2.0
		x = x - self.xopt

		x = x[:, self.P]
		f = 0
		idx_u = 0
		for i in range(len(self.s)):
			si = self.s[i]
			idx_l = idx_u
			idx_u = idx_l + si
			if si == 25:
				Ri = self.R25
			elif si == 50:
				Ri = self.R50
			elif si == 100:
				Ri = self.R100

			xi = x[:, idx_l:idx_u]
			xi = torch.sum(Ri * xi.unsqueeze(1), dim=2)
			#xi = self.T_diag(10.0, xi.size(1))*self.T_asy(self.T_osz(xi), 0.2)
			f = f + self.Rastrigin_func(xi) * self.w[i]

		xi = x[:, idx_u:]
		#xi = self.T_diag(10.0, xi.size(1))*self.T_asy(self.T_osz(xi), 0.2)
		f = f + self.Rastrigin_func(xi) 

		if is_norm:
			return f/self.norm
		else:
			return f


	def f4(self, x, is_norm=True):
		x_u = 32
		x_l = -32

		x = x * (x_u - x_l)/2.0 + (x_u + x_l)/2.0
		x = x - self.xopt

		x = x[:, self.P]
		f = 0
		idx_u = 0
		for i in range(len(self.s)):
			si = self.s[i]
			idx_l = idx_u
			idx_u = idx_l + si
			if si == 25:
				Ri = self.R25
			elif si == 50:
				Ri = self.R50
			elif si == 100:
				Ri = self.R100

			xi = x[:, idx_l:idx_u]
			xi = torch.sum(Ri * xi.unsqueeze(1), dim=2)
			#xi = self.T_diag(10.0, xi.size(1))*self.T_asy(self.T_osz(xi), 0.2)
			f = f + self.Ackley_func(xi) * self.w[i]

		xi = x[:, idx_u:]
		#xi = self.T_diag(10.0, xi.size(1))*self.T_asy(self.T_osz(xi), 0.2)
		f = f + self.Ackley_func(xi) 

		if is_norm:
			return f/self.norm
		else:
			return f


	

	


	def f5(self, x, is_norm=True):
		x_u = 5
		x_l = -5

		x = x * (x_u - x_l)/2.0 + (x_u + x_l)/2.0
		x = x - self.xopt
		x = x[:, self.P]

		f = 0
		idx_u = 0
		for i in range(len(self.s)):
			si = self.s[i]
			idx_l = idx_u
			idx_u = idx_l + si
			if si == 25:
				Ri = self.R25
			elif si == 50:
				Ri = self.R50
			elif si == 100:
				Ri = self.R100

			xi = x[:, idx_l:idx_u]
			xi = torch.sum(Ri * xi.unsqueeze(1), dim=2)
			f = f + self.Rastrigin_func(xi) * self.w[i]

		if is_norm:
			return f/self.norm
		else:
			return f


	def f6(self, x, is_norm=True):
		x_u = 32
		x_l = -32

		x = x * (x_u - x_l)/2.0 + (x_u + x_l)/2.0
		x = x - self.xopt
		x = x[:, self.P]

		f = 0
		idx_u = 0
		for i in range(len(self.s)):
			si = self.s[i]
			idx_l = idx_u
			idx_u = idx_l + si
			if si == 25:
				Ri = self.R25
			elif si == 50:
				Ri = self.R50
			elif si == 100:
				Ri = self.R100

			xi = x[:, idx_l:idx_u]
			xi = torch.sum(Ri * xi.unsqueeze(1), dim=2)
			f = f + self.Ackley_func(xi) * self.w[i]

		if is_norm:
			return f/self.norm
		else:
			return f

	