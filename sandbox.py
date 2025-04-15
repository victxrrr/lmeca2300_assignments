import numpy as np

a = np.array([1, 2, 3, 4, 5])

mask = (a >= 3)
a[mask] = 10

a[~mask] = 2*a[~mask]

print(a)


def generate_rect(center, n_height, n_width):
    length = lambda_/10
    hh = n_height * length/2
    ww = n_width * length/2
    A = [center[0] - ww, center[1] - hh]
    B = [center[0] + ww, center[1] - hh]
    C = [center[0] + ww, center[1] + hh]
    D = [center[0] - ww, center[1] + hh]

    rect = []
    for i in range(n_width):
        rect.append(Segment(
            [A[0] + i * length, A[1]], [A[0] + (i+1) * length, A[1]]
        ))
    for i in range(n_height):
        rect.append(Segment(
            [B[0], B[1] + i * length], [B[0], B[1] + (i+1) * length]
        ))
    for i in range(n_width):
        rect.append(Segment(
            [C[0] - i * length, C[1]], [C[0] - (i+1) * length, C[1]]
        ))
    for i in range(n_height):
        rect.append(Segment(
            [D[0], D[1] - i * length], [D[0], D[1] - (i+1) * length]
        ))

    return rect

groups = [
    generate_rect([-lambda_, lambda_], 1, 2),
    generate_rect([-1.3*lambda_, lambda_], 1, 1),
    generate_rect([-1.15*lambda_, 0.8*lambda_], 1, 3)
]

### show geometry
for g in groups:
    for s in g:
        plt.plot([s.r1[0], s.r2[0]], [s.r1[1], s.r2[1]], "b.-")
    plt.axis("equal")
    plt.xlabel("$x$"); plt.ylabel("$y$")
plt.grid()
plt.show()

def generate_foam(N, phi):
    """
    @param N: number of wedges
    @param phi: wedge angle in degrees
    """

    wedge_side = lambda_/10
    width = wedge_side/1
    phi = np.deg2rad(phi)
    wedge_base = wedge_side * np.sqrt(2 * (1 - np.cos(phi)))

    shape = [Segment([0,0], [width,0])]

    for i in range(N):
        A = [width, i*wedge_base]
        B = [A[0] + wedge_side * np.cos(phi/2), A[1] + wedge_side * np.sin(phi/2)]
        C = [A[0], A[1] + wedge_base]
        shape.append(Segment(A, B))
        shape.append(Segment(B, C))

    shape.append(Segment([width, N*wedge_base], [0, N*wedge_base]))
    shape.append(Segment([0, N*wedge_base], [0,0]))

    return shape

theShape = generate_foam(10, 60)

### show geometry
for s in theShape:
    plt.plot([s.r1[0], s.r2[0]], [s.r1[1], s.r2[1]], "b.-")
plt.axis("equal")
plt.xlabel("$x$"); plt.ylabel("$y$")
plt.grid()
plt.show()