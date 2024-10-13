import math

# 1. Parsing the system of Equations

def parse_equations(filename):
    A = []  # to store the coefficients of x, y, z
    B = []  # to store the constants on the right-hand side

    with open(filename, 'r') as file:
        for line in file:
            # split the line into parts based on spaces
            parts = line.split()

            # initialize the coefficients for x, y, z to 0
            coefficients = [0, 0, 0]

            # variable mapping: [x, y, z] -> index [0, 1, 2]
            variable_indices = {'x': 0, 'y': 1, 'z': 2}

            i = 0
            sign = 1
            for part in parts:
                if part == '=':
                    B.append(int(parts[parts.index('=') + 1]))
                    break
                elif part == '+':
                    sign = 1
                elif part == '-':
                    sign = -1
                else:
                    if 'x' in part or 'y' in part or 'z' in part:
                        # find out which variable it is
                        variable = part[-1]
                        coeff = part[:-1]

                        if coeff == '' or coeff == '+':
                            coeff = 1  # implicit positive coefficient
                        elif coeff == '-':
                            coeff = -1  # implicit negative coefficient
                        else:
                            coeff = int(coeff)

                        # apply the sign and store the coefficient
                        coefficients[variable_indices[variable]] = sign * coeff

            A.append(coefficients)

    return A, B

# Determinant of a 3x3 Matrix

def determinant_3x3(A):
    # Extract values from matrix A
    a11, a12, a13 = A[0]
    a21, a22, a23 = A[1]
    a31, a32, a33 = A[2]

    # Calculate determinant using the formula
    det = (a11 * (a22 * a33 - a23 * a32)
           - a12 * (a21 * a33 - a23 * a31)
           + a13 * (a21 * a32 - a22 * a31))

    return det

# Trace of a 3x3 matrix

def trace_3x3(A):
    return A[0][0] + A[1][1] + A[2][2]

# Euclidean Norm of a vector

def vector_norm(B):
    return math.sqrt(B[0]**2 + B[1]**2 + B[2]**2)

# Transpose of a 3x3 matrix

def transpose_3x3(A):
    return [[A[0][0], A[1][0], A[2][0]],
            [A[0][1], A[1][1], A[2][1]],
            [A[0][2], A[1][2], A[2][2]]]

# Matrix-vector multiplication

def matrix_vector_multiply(A, B):
    # Multiply each row of matrix A by vector B
    result = [
        A[0][0] * B[0] + A[0][1] * B[1] + A[0][2] * B[2],
        A[1][0] * B[0] + A[1][1] * B[1] + A[1][2] * B[2],
        A[2][0] * B[0] + A[2][1] * B[1] + A[2][2] * B[2]
    ]
    return result


# Parsing the system of equations
filename = 'input.txt'
A, B = parse_equations(filename)

print("Matrix A (coefficients):")
for row in A:
    print(row)

print("\nVector B (constants):")
print(B)

# Determinant
det_A = determinant_3x3(A)
print(f"Determinant of A: {det_A}")

# Trace
trace_A = trace_3x3(A)
print(f"Trace of A: {trace_A}")

# Vector norm
norm_B = vector_norm(B)
print(f"Norm of vector B: {norm_B}")

# Transpose
transpose_A = transpose_3x3(A)
print("Transpose of A:")
for row in transpose_A:
    print(row)

# Matrix-vector multiplication
result_vector = matrix_vector_multiply(A, B)
print("Matrix-vector multiplication result (A * B):", result_vector)
