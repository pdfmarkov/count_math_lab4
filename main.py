import math
import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

##############################
#       MATRIX LOGIC         #
##############################

# Getting the matrix from user
def get_matrix():
    new_matrix = []
    # print('Welcome to the Gauss calculator!')
    # print('Please, tell me, where is your matrix?')
    # print('Write 1 if you want to write it by yourself')
    # print('Write 0 if you want to get matrix from file')
    answer = int(input())
    if answer == 1:
        # print('Great! So, how many rows do you have?')
        rows = int(input())
        # print('Good job! Let\'s move forward')
        if rows != 1:
            for i in range(rows):
                a = []
                for j in range(rows + 1):
                    # print('Write elem: row', i + 1, ', column:', j + 1)
                    a.append(float(input()))
                new_matrix.append(a)
            # print('Thank you! Your matrix is here...')
            print(new_matrix)
            return new_matrix
        else:
            # print('Oh, matrix should be 2x2, 3x3 or bigger! Please, try again!')
            return get_matrix()
    elif answer == 0:
        # print('It\'s so great! Please, write the name of your file!')
        return get_matrix_from_file(input())
    else:
        # print('Oh, your answer is broken! Please, try again!')
        return get_matrix()


# Getting matrix from the file by filename
def get_matrix_from_file(filename):
    with open(filename) as f:
        matrix_from_file = [list(map(float, row.split())) for row in f.readlines()]
    # print('Matrix:')
    print_matrix(matrix_from_file, 5)
    return matrix_from_file


# Realising a Gauss's Method
def do_gauss_method(input_matrix):
    check_square_matrix(input_matrix)
    # print('Let\'s do Gauss method')
    length_of_matrix = len(input_matrix)
    do_triangle_matrix(input_matrix)
    is_singular(input_matrix)
    input_answer_matrix = [0 for i in range(length_of_matrix)]
    for k in range(length_of_matrix - 1, -1, -1):
        input_answer_matrix[k] = (input_matrix[k][-1] - sum(
            [input_matrix[k][j] * input_answer_matrix[j] for j in range(k + 1, length_of_matrix)])) / input_matrix[k][k]
    # print('FINALLY!')
    # print('ヽ(⌐■_■)ノ♪♬')
    # print('ANSWERS!')
    # for i in range(len(input_answer_matrix)):
    #     print('x[', i + 1, '] =', "%5.3f" % input_answer_matrix[i])
    return input_answer_matrix


# Checking the matrix
def check_square_matrix(input_matrix):
    # print('Check if matrix is square (and extended) AND has solutions')
    for i in range(len(input_matrix)):
        if len(input_matrix) + 1 != len(input_matrix[i]):
            raise Exception('ERROR: The size of matrix isn\'t correct')
        count = 0
        for j in range(len(input_matrix[i]) - 1):
            if input_matrix[i][j] == 0:
                count += 1
        if count == len(input_matrix[i]) - 1:
            raise Exception('ERROR: The matrix has no solutions')
    # print('All is okay!')


def do_triangle_matrix(input_matrix):
    # print('Let\'s create a Triangle Matrix')
    length_of_matrix = len(input_matrix)  # = number of rows
    for k in range(length_of_matrix - 1):
        # print('Iteration №', k + 1)
        # print('Matrix was...')
        # print_matrix(input_matrix, 5)
        get_max_element_in_column(input_matrix, k)
        # print('Matrix become...')
        # print_matrix(input_matrix, 5)
        # print('Do some math magic...')
        # print('╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ *wibbly wobbly timey wimey*')
        # print('**TA DA**')
        for i in range(k + 1, length_of_matrix):
            div = input_matrix[i][k] / input_matrix[k][k]
            input_matrix[i][-1] -= div * input_matrix[k][-1]
            for j in range(k, length_of_matrix):
                input_matrix[i][j] -= div * input_matrix[k][j]
        # print_matrix(input_matrix, 5)
    return length_of_matrix


# Checking if matrix is singular (вырожденная)
def is_singular(input_matrix):
    # print('Check if matrix is singular (вырожденная)')
    if count_determinant_for_square_matrix(input_matrix) == 0:
        raise Exception('ERROR: Your matrix is singular (det ~ 0)')
    # else:
    #     print('OKAY, It isn\'t')


# Searching for the main element in the column
def get_max_element_in_column(input_matrix, number_of_column):
    max_element = input_matrix[number_of_column][number_of_column]
    max_row = number_of_column
    for j in range(number_of_column + 1, len(input_matrix)):
        if abs(input_matrix[j][number_of_column]) > abs(max_element):
            max_element = input_matrix[j][number_of_column]
            max_row = j
    if max_row != number_of_column:
        input_matrix[number_of_column], input_matrix[max_row] = input_matrix[max_row], input_matrix[number_of_column]
    # print('The max element between not fixed rows is', "%.4f" % max_element, 'in row', max_row + 1)
    return input_matrix


# Printing matrix in the comfortable view
def print_matrix(input_matrix, decimals):
    cprint(tabulate(input_matrix,
                    tablefmt="fancy_grid", floatfmt="2.5f"), 'cyan')


# Create residual vector (вектор невязок)
def do_residual_vector(input_matrix, input_answer_matrix):
    big_matrix = []
    little_matrix = []
    for i in range(len(input_matrix)):
        big_matrix.append(input_matrix[i][0:len(input_matrix)])
        little_matrix.append(input_matrix[i][len(input_matrix):])
    x_matrix = input_answer_matrix
    temp = [0 for i in range(len(input_matrix))]
    residual_vector = [0 for i in range(len(input_matrix))]
    # print('Good job! But...')
    # print('Let\'s find residual vector!')
    # print('Residual vector:')
    for i in range(len(big_matrix)):
        temp[i] = 0
        for j in range(len(big_matrix)):
            temp[i] += x_matrix[j] * big_matrix[i][j]
        residual_vector[i] = temp[i] - little_matrix[i][0]
        # print('r[', i + 1, '] =', residual_vector[i], end='\n')


# Counting the determinant of the out matrix
def count_determinant_for_square_matrix(input_matrix):
    determinant = 1
    for i in range(len(input_matrix)):
        determinant *= input_matrix[i][i]
    # print('Determinant of matrix =', round(determinant, 5))
    return round(determinant, 5)


##############################
#       APPROXIMATION        #
##############################

# Getting x and f(x) from the file by filename
def get_xy_from_file():
    cprint('Please, write the name of your file!', 'yellow')
    filename = input()
    with open(filename) as f:
        this_xy = [list(map(float, row.split())) for row in f.readlines()]
        this_xy[0].insert(0, "X")
        this_xy[1].insert(0, "Y")
    cprint('\nFunction:', 'cyan', attrs=['bold'])
    print_matrix(this_xy, 5)
    this_xy[0].pop(0)
    this_xy[1].pop(0)
    return this_xy


def lineal(input_xy):
    x = input_xy[0]
    y = input_xy[1]
    sx = 0
    sxx = 0
    sy = 0
    sxy = 0
    for i in range(len(x)):
        sx += x[i]
        sxx += x[i] ** 2
        sy += y[i]
        sxy += x[i] * y[i]
    param_matrix = [[sxx, sx, sxy], [sx, len(x), sy]]
    answer_matrix = do_gauss_method(param_matrix)
    print()
    text = '\tLINEAR: y = ' + str(answer_matrix[0]) + ' * x + ' + str(answer_matrix[1])
    cprint(text, 'green', attrs=['bold'])
    s = 0
    for i in range(len(x)):
        s += (answer_matrix[0] * x[i] + answer_matrix[1] - y[i]) ** 2
    text = '\t\tМера отклонения: ' + str(s)
    cprint(text, 'green')
    text = '\t\tСреднеквадратичное отклонение: ' + str(math.sqrt(s / len(x)))
    cprint(text, 'green')
    up = 0
    sum_of_lineal_func = 0
    squares = 0
    for i in range(len(x)):
        up += (y[i] - (answer_matrix[0] * x[i] + answer_matrix[1])) ** 2
        sum_of_lineal_func += answer_matrix[0] * x[i] + answer_matrix[1]
        squares += (answer_matrix[0] * x[i] + answer_matrix[1]) ** 2
    text = '\t\tДостоверность аппроксимации: ' + str(1 - (up / (squares - (sum_of_lineal_func ** 2) / len(x))))
    cprint(text, 'green')
    middle_x = 0
    middle_y = 0
    for i in range(len(x)):
        middle_x += x[i]
        middle_y += y[i]
    middle_x = middle_x / len(x)
    middle_y = middle_y / len(y)
    upper = 0
    under_x = 0
    under_y = 0
    for i in range(len(x)):
        upper += (x[i] - middle_x) * (y[i] - middle_y)
        under_x += (x[i] - middle_x) ** 2
        under_y += (y[i] - middle_y) ** 2
    r = upper / math.sqrt(under_x * under_y)
    text = '\t\tКоэффициент Пирсона: ' + str(r)
    cprint(text, 'green')
    answer_matrix.append(math.sqrt(s / len(x)))
    return answer_matrix


def square(input_xy):
    x = input_xy[0]
    y = input_xy[1]
    sx = 0
    sxx = 0
    sxxx = 0
    sxxxx = 0
    sy = 0
    sxy = 0
    sxxy = 0
    for i in range(len(x)):
        sx += x[i]
        sxx += x[i] ** 2
        sxxx += x[i] ** 3
        sxxxx += x[i] ** 4
        sy += y[i]
        sxy += x[i] * y[i]
        sxxy += x[i] ** 2 * y[i]
    param_matrix = [[len(x), sx, sxx, sy], [sx, sxx, sxxx, sxy], [sxx, sxxx, sxxxx, sxxy]]
    answer_matrix = do_gauss_method(param_matrix)
    print()
    text = '\tSQUARE: y = ' + str(answer_matrix[0]) + ' + ' + str(answer_matrix[1]) + ' * x + ' + str(
        answer_matrix[2]) + ' * x ^ 2'
    cprint(text, 'green', attrs=['bold'])
    s = 0
    for i in range(len(x)):
        s += (answer_matrix[0] + answer_matrix[1] * x[i] + answer_matrix[2] * x[i] ** 2 - y[i]) ** 2
    text = '\t\tМера отклонения: ' + str(s)
    cprint(text, 'green')
    text = '\t\tСреднеквадратичное отклонение: ' + str(math.sqrt(s / len(x)))
    cprint(text, 'green')
    up = 0
    sum_of_lineal_func = 0
    squares = 0
    for i in range(len(x)):
        up += (y[i] - (answer_matrix[0] + answer_matrix[1] * x[i] + answer_matrix[2] * x[i] ** 2)) ** 2
        sum_of_lineal_func += answer_matrix[0] + answer_matrix[1] * x[i] + answer_matrix[2] * x[i] ** 2
        squares += (answer_matrix[0] + answer_matrix[1] * x[i] + answer_matrix[2] * x[i] ** 2) ** 2
    text = '\t\tДостоверность аппроксимации: ' + str(1 - (up / (squares - (sum_of_lineal_func ** 2) / len(x))))
    cprint(text, 'green')
    answer_matrix.append(math.sqrt(s / len(x)))
    return answer_matrix


def exponent(input_xy):
    x = input_xy[0]
    y = input_xy[1]
    sx = 0
    sxx = 0
    sy = 0
    sxy = 0
    for i in range(len(x)):
        if y[i] > 0:
            sx += x[i]
            sxx += x[i] ** 2
            sy += math.log(y[i], math.e)
            sxy += x[i] * math.log(y[i], math.e)
        else:
            print()
            cprint('\tEXPONENT method can\'t be used', 'red', attrs=['bold'])
            return
    param_matrix = [[len(x), sx, sy], [sx, sxx, sxy]]
    answer_matrix = do_gauss_method(param_matrix)
    print()
    text = '\tEXPONENT: y = ' + str(math.e ** answer_matrix[0]) + ' * e ^ ( ' + str(answer_matrix[1]) + ' * x )'
    cprint(text, 'green', attrs=['bold'])
    s = 0
    for i in range(len(x)):
        s += ((math.e ** answer_matrix[0] * math.e ** (answer_matrix[1] * x[i])) - y[i]) ** 2
    text = '\t\tМера отклонения: ' + str(s)
    cprint(text, 'green')
    text = '\t\tСреднеквадратичное отклонение: ' + str(math.sqrt(s / len(x)))
    cprint(text, 'green')
    up = 0
    sum_of_lineal_func = 0
    squares = 0
    for i in range(len(x)):
        up += (y[i] - (math.e ** answer_matrix[0] * math.e ** (answer_matrix[1] * x[i]))) ** 2
        sum_of_lineal_func += (math.e ** answer_matrix[0] * math.e ** (answer_matrix[1] * x[i]))
        squares += (math.e ** answer_matrix[0] * math.e ** (answer_matrix[1] * x[i])) ** 2
    text = '\t\tДостоверность аппроксимации: ' + str(1 - (up / (squares - (sum_of_lineal_func ** 2) / len(x))))
    cprint(text, 'green')
    answer_matrix.append(math.sqrt(s / len(x)))
    return answer_matrix


def degree(input_xy):
    x = input_xy[0]
    y = input_xy[1]
    sx = 0
    sxx = 0
    sy = 0
    sxy = 0
    for i in range(len(x)):
        if y[i] > 0 and x[i] > 0:
            sx += math.log(x[i], math.e)
            sxx += math.log(x[i], math.e) ** 2
            sy += math.log(y[i], math.e)
            sxy += math.log(x[i], math.e) * math.log(y[i], math.e)
        else:
            print()
            cprint('\tDEGREE method can\'t be used', 'red', attrs=['bold'])
            return
    param_matrix = [[len(x), sx, sy], [sx, sxx, sxy]]
    answer_matrix = do_gauss_method(param_matrix)
    print()
    text = '\tDEGREE: y = ' + str(math.e ** answer_matrix[0]) + ' * x ^ ( ' + str(answer_matrix[1]) + ' )'
    cprint(text, 'green', attrs=['bold'])
    s = 0
    for i in range(len(x)):
        s += ((math.e ** answer_matrix[0] * x[i] ** (answer_matrix[1])) - y[i]) ** 2
    text = '\t\tМера отклонения: ' + str(s)
    cprint(text, 'green')
    text = '\t\tСреднеквадратичное отклонение: ' + str(math.sqrt(s / len(x)))
    cprint(text, 'green')
    up = 0
    sum_of_lineal_func = 0
    squares = 0
    for i in range(len(x)):
        up += (y[i] - (math.e ** answer_matrix[0] * x[i] ** (answer_matrix[1]))) ** 2
        sum_of_lineal_func += (math.e ** answer_matrix[0] * x[i] ** (answer_matrix[1]))
        squares += (math.e ** answer_matrix[0] * x[i] ** (answer_matrix[1])) ** 2
    text = '\t\tДостоверность аппроксимации: ' + str(1 - (up / (squares - (sum_of_lineal_func ** 2) / len(x))))
    cprint(text, 'green')
    answer_matrix.append(math.sqrt(s / len(x)))
    return answer_matrix


def log(input_xy):
    x = input_xy[0]
    y = input_xy[1]
    sx = 0
    sxx = 0
    sy = 0
    sxy = 0
    for i in range(len(x)):
        if x[i] > 0:
            sx += math.log(x[i], math.e)
            sxx += math.log(x[i], math.e) ** 2
            sy += y[i]
            sxy += math.log(x[i], math.e) * y[i]
        else:
            print()
            cprint('\tLOGARIFM method can\'t be used', 'red', attrs=['bold'])
            return
    param_matrix = [[len(x), sx, sy], [sx, sxx, sxy]]
    answer_matrix = do_gauss_method(param_matrix)
    print()
    text = '\tLOGARIFM: y = ' + str(answer_matrix[1]) + ' * ln(x) + ' + str(answer_matrix[0])
    cprint(text, 'green', attrs=['bold'])
    s = 0
    for i in range(len(x)):
        s += ((answer_matrix[1] * math.log(x[i], math.e) + answer_matrix[0]) - y[i]) ** 2
    text = '\t\tМера отклонения: ' + str(s)
    cprint(text, 'green')
    text = '\t\tСреднеквадратичное отклонение: ' + str(math.sqrt(s / len(x)))
    cprint(text, 'green')
    up = 0
    sum_of_lineal_func = 0
    squares = 0
    for i in range(len(x)):
        up += (y[i] - (answer_matrix[1] * math.log(x[i], math.e) + answer_matrix[0])) ** 2
        sum_of_lineal_func += (answer_matrix[1] * math.log(x[i], math.e) + answer_matrix[0])
        squares += (answer_matrix[1] * math.log(x[i], math.e) + answer_matrix[0]) ** 2
    text = '\t\tДостоверность аппроксимации: ' + str(1 - (up / (squares - (sum_of_lineal_func ** 2) / len(x))))
    cprint(text, 'green')
    answer_matrix.append(math.sqrt(s / len(x)))
    return answer_matrix


def print_table(input_xy, input_lineal_matrix, input_square_matrix, input_exponent_matrix, input_degree_matrix,
                input_log_matrix):
    print()
    new_table = []
    x = input_xy[0]
    y = input_xy[1]
    for i in range(len(x)):
        if input_exponent_matrix is None and input_log_matrix is None:
            new_table.append([i + 1, x[i], y[i], input_lineal_matrix[0] * x[i] + input_lineal_matrix[1],
                              input_square_matrix[0] + input_square_matrix[1] * x[i] + input_square_matrix[2] * x[i] ** 2])
        elif input_exponent_matrix is None:
            new_table.append([i + 1, x[i], y[i], input_lineal_matrix[0] * x[i] + input_lineal_matrix[1],
                              input_square_matrix[0] + input_square_matrix[1] * x[i] + input_square_matrix[2] * x[i] ** 2,
                              input_log_matrix[1] * math.log(x[i], math.e) + input_log_matrix[0]])
        elif input_log_matrix is None:
            new_table.append([i + 1, x[i], y[i], input_lineal_matrix[0] * x[i] + input_lineal_matrix[1],
                              input_square_matrix[0] + input_square_matrix[1] * x[i] + input_square_matrix[2] * x[i] ** 2,
                              math.e ** input_exponent_matrix[0] * math.e ** (input_exponent_matrix[1] * x[i])])
        else:
            new_table.append([i + 1, x[i], y[i], input_lineal_matrix[0] * x[i] + input_lineal_matrix[1],
                          input_square_matrix[0] + input_square_matrix[1] * x[i] + input_square_matrix[2] * x[i] ** 2,
                          math.e ** input_exponent_matrix[0] * math.e ** (input_exponent_matrix[1] * x[i]),
                          math.e ** input_degree_matrix[0] * x[i] ** (input_degree_matrix[1]),
                          input_log_matrix[1] * math.log(x[i], math.e) + input_log_matrix[0]])
    cprint('\nTable:', 'cyan', attrs=['bold'])
    if input_exponent_matrix is None and input_log_matrix is None:
        cprint(tabulate(new_table, headers=["№", "x", "f(x)", "LIN", "SQR"],
                        tablefmt="fancy_grid", floatfmt="2.5f"), 'cyan')
    elif input_exponent_matrix is None:
        cprint(tabulate(new_table, headers=["№", "x", "f(x)", "LIN", "SQR", "LOG"],
                        tablefmt="fancy_grid", floatfmt="2.5f"), 'cyan')
    elif input_log_matrix is None:
        cprint(tabulate(new_table, headers=["№", "x", "f(x)", "LIN", "SQR", "EXP"],
                        tablefmt="fancy_grid", floatfmt="2.5f"), 'cyan')
    else:
        cprint(tabulate(new_table, headers=["№", "x", "f(x)", "LIN", "SQR", "EXP", "DEG", "LOG"],
                 tablefmt="fancy_grid", floatfmt="2.5f"), 'cyan')


def choose_best_approximation(input_lineal_matrix, input_square_matrix, input_exponent_matrix, input_degree_matrix,
                              input_log_matrix):
    print()
    cprint('Best approximation:', 'green', attrs=['bold'])
    print()

    if input_exponent_matrix is None and input_log_matrix is None:
        min_sqr = min(input_lineal_matrix[2], input_square_matrix[3])
    elif input_exponent_matrix is None:
        min_sqr = min(input_lineal_matrix[2], input_square_matrix[3], input_log_matrix[2])
    elif input_log_matrix is None:
        min_sqr = min(input_lineal_matrix[2], input_square_matrix[3], input_exponent_matrix[2])
    else:
        min_sqr = min(input_lineal_matrix[2], input_square_matrix[3], input_exponent_matrix[2], input_degree_matrix[2],
                      input_log_matrix[2])

    if min_sqr == input_lineal_matrix[2]:
        text = '\tLINEAR: y = ' + str(input_lineal_matrix[0]) + ' * x + ' + str(input_lineal_matrix[1])
        cprint(text, 'green', attrs=['bold'])
    elif min_sqr == input_square_matrix[3]:
        text = '\tSQUARE: y = ' + str(input_square_matrix[0]) + ' + ' + str(input_square_matrix[1]) + ' * x + ' + str(
            input_square_matrix[2]) + ' * x ^ 2'
        cprint(text, 'green', attrs=['bold'])
    elif input_exponent_matrix is not None and min_sqr == input_exponent_matrix[2]:
        text = '\tEXPONENT: y = ' + str(math.e ** input_exponent_matrix[0]) + ' * e ^ ( ' + str(
            input_exponent_matrix[1]) + ' * x )'
        cprint(text, 'green', attrs=['bold'])
    elif input_log_matrix is not None and min_sqr == input_log_matrix[2]:
        text = '\tLOGARIFM: y = ' + str(input_log_matrix[1]) + ' * ln(x) + ' + str(input_log_matrix[0])
        cprint(text, 'green', attrs=['bold'])
    elif min_sqr == input_degree_matrix[2]:
        text = '\tDEGREE: y = ' + str(math.e ** input_degree_matrix[0]) + ' * x ^ ( ' + str(
            input_degree_matrix[1]) + ' )'
        cprint(text, 'green', attrs=['bold'])



def create_graph(input_xy, input_lineal_matrix, input_square_matrix, input_exponent_matrix, input_degree_matrix,
                 input_log_matrix):
    try:
        ax = plt.gca()
        plt.grid()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        minimum = input_xy[0][0]
        maximum = input_xy[0][-1]
        x = np.linspace(minimum - ((maximum - minimum) / 20), maximum + ((maximum - minimum) / 20), 100)
        plt.title('y = f(x)')
        plt.plot(x, input_lineal_matrix[0] * x + input_lineal_matrix[1], color="g", linewidth=1, label='LIN')
        plt.plot(x, input_square_matrix[0] + input_square_matrix[1] * x + input_square_matrix[2] * x ** 2, color="b",
                 linewidth=1, label='SQR')
        if input_exponent_matrix is not None:
            plt.plot(x, math.e ** input_exponent_matrix[0] * math.e ** (input_exponent_matrix[1] * x), color="c",
                 linewidth=1, label='EXP')
        if input_degree_matrix is not None:
            plt.plot(x, math.e ** input_degree_matrix[0] * x ** (input_degree_matrix[1]), color="m", linewidth=1,
                 label='DEG')
        if input_log_matrix is not None:
            plt.plot(x, input_log_matrix[1] * np.log(x) + input_log_matrix[0], color="y", linewidth=1, label='LOG')
        plt.plot(x, 0 * x, color="black", linewidth=1)
        for i in range(len(input_xy[0])):
            plt.scatter(input_xy[0][i], input_xy[1][i], color="r", s=30)
        plt.legend()
        plt.show()

        fig, axs = plt.subplots(5, sharex=True, sharey=True)
        fig.suptitle('y = f(x)')
        axs[0].plot(x, input_lineal_matrix[0] * x + input_lineal_matrix[1], color="g", linewidth=1, label='LIN')
        axs[1].plot(x, input_square_matrix[0] + input_square_matrix[1] * x + input_square_matrix[2] * x ** 2, color="b",
                    linewidth=1, label='SQR')
        if input_exponent_matrix is not None:
            axs[2].plot(x, math.e ** input_exponent_matrix[0] * math.e ** (input_exponent_matrix[1] * x), color="c",
                    linewidth=1, label='EXP')
        if input_degree_matrix is not None:
            axs[3].plot(x, math.e ** input_degree_matrix[0] * x ** (input_degree_matrix[1]), color="m", linewidth=1,
                    label='DEG')
        if input_log_matrix is not None:
            axs[4].plot(x, input_log_matrix[1] * np.log(x) + input_log_matrix[0], color="y", linewidth=1, label='LOG')
        for i in range(5):
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].grid()
            if not ((i == 2 and input_exponent_matrix is None) or (i == 3 and input_degree_matrix is None) or (i == 4 and input_log_matrix is None)):
                axs[i].legend()
                for j in range(len(input_xy[0])):
                    axs[i].scatter(input_xy[0][j], input_xy[1][j], color="r", s=20)
        plt.show()
        del x
    except ValueError:
        return
    except ZeroDivisionError:
        return
    except OverflowError:
        return


try:
    cprint('! WELCOME TO THE APPROXIMATION CALCULATOR !\n', 'yellow', attrs=['bold'])

    xy = get_xy_from_file()
    cprint('\nApproximation functions:', 'green', attrs=['bold'])
    lineal_matrix = lineal(xy)
    square_matrix = square(xy)
    exponent_matrix = exponent(xy)
    degree_matrix = degree(xy)
    log_matrix = log(xy)
    print_table(xy, lineal_matrix, square_matrix, exponent_matrix, degree_matrix, log_matrix)
    choose_best_approximation(lineal_matrix, square_matrix, exponent_matrix, degree_matrix, log_matrix)
    create_graph(xy, lineal_matrix, square_matrix, exponent_matrix, degree_matrix, log_matrix)
except Exception as ex:
    template = "Oh, you've got an exception! What a pity! So, the problem is...\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print(message)
