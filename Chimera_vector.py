import numpy as np

def CHIMERA_Vector(params, domain):
    M = np.zeros((9, domain[1] + 1))

    # Definicion de condiciones iniciales
    M[0, 0] = params[0]
    M[1, 0] = params[1]
    M[2, 0] = params[2]
    M[3, 0] = params[3]
    M[4, 0] = params[4]
    M[5, 0] = params[5]
    M[6, 0] = params[1]
    M[7, 0] = params[1]
    M[8, 0] = np.sum(M[0:3, 0])

    # Definicion de parametros
    gamma = params[6]
    mu_m = params[7]
    mu_h = params[8]
    z = params[9]
    r = params[10]
    C = params[11]
    beta_mh = params[12]
    beta_hm = params[13]
    nu_m = params[14]
    nu_h = params[15]

    # Poblacion total
    N_M = np.zeros(domain[1] + 1)

    # simulacion del modelo
    for i in range(domain[0], domain[1]):
        # Poblacion total de humanos
        N_H = np.sum(M[0:3, i])

        # Definicion de probabilidades de infeccion
        aleph_H = 1 + nu_h * M[1, i] / (M[0, i] + M[1, i])
        aleph_M = 1 + nu_m * M[4, i] / (M[3, i] + M[4, i])
        psi = beta_hm * (1 - (1 - (M[1, i] / N_H) ** aleph_M) ** z)
        red_m = z * M[4, i] * (M[0, i] / N_H) ** aleph_H
        if M[0, i] >= 1:
            phi = beta_mh * (1 - (1 - (1 / M[0, i])) ** red_m)
        else:
            phi = 0

        # Ecuaciones para poblacion humana
        M[0, i + 1] = M[0, i] * (1 - phi) * (1 - mu_h) + (M[0, i] + M[1, i] + M[2, i]) * mu_h
        M[1, i + 1] = M[0, i] * phi * (1 - mu_h) + M[1, i] * (1 - gamma) * (1 - mu_h)
        M[2, i + 1] = M[2, i] * (1 - mu_h) + M[1, i] * gamma * (1 - mu_h)

        M[6, i + 1] = M[0, i] * phi * (1 - mu_h)  # casos instantaneos
        M[7, i + 1] = M[7, i] + M[0, i] * phi * (1 - mu_h)  # casos acumulados

        # Ecuaciones para poblacion de mosquitos
        N_M[i] = np.sum(M[3:5, i])

        if i > 2:
            A = r * N_M[i - 2] * np.exp((1 - N_M[i] / C))
        else:
            A = r * N_M[1] * np.exp((1 - N_M[1] / C))

        M[3, i + 1] = M[3, i] * (1 - psi) * (1 - mu_m) + A
        M[4, i + 1] = M[3, i] * psi * (1 - mu_m) + M[4, i] * (1 - mu_m)
        M[5, i + 1] = M[4, i] * mu_m + M[3, i] * mu_m

        M[8, i + 1] = N_H

    # Salidas del modelo
    sol = {
        'x': np.arange(domain[0], domain[1] + 1),
        'y': M
    }

    return sol