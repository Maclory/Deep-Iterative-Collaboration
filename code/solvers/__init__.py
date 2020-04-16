
def create_solver(opt):
    if opt['mode'] == 'sr_align':
        from .SRLandmarkSolver import SRLandmarkSolver
        solver = SRLandmarkSolver(opt)
    elif opt['mode'] == 'sr_align_gan':
        from .SRLandmarkGANSolver import SRLandmarkGANSolver
        solver = SRLandmarkGANSolver(opt)
    else:
        raise NotImplementedError('Solver mode %s not implemented!' % opt['mode'])

    return solver
