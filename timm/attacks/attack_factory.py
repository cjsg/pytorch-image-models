import foolbox as fb

def create_attack(name, steps=None, random_start=None, **kwargs):
    if name == 'fgsm':
        random_start = random_start or False
        attack = fb.attacks.FGSM(random_start=random_start, **kwargs)
    elif name == 'fgm':
        random_start = random_start or False
        attack = fb.attacks.FGM(random_start=random_start, **kwargs)
    elif name == 'l2pgd':
        steps = steps or 10
        random_start = random_start or True
        attack = fb.attacks.L2PGD(steps=steps, random_start=random_start, **kwargs)
    elif name in {'pgd', 'linfpgd'}:
        steps = steps or 10
        random_start = random_start or True
        attack = fb.attacks.LinfPGD(steps=steps, random_start=random_start, **kwargs)
    else:
        raise ValueError(f'Unkonwn attack {name}.')
    return attack
