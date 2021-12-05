import re
import numpy as np
import fractions
from io import StringIO
from fractions import Fraction

np.set_printoptions(formatter={'all': lambda x: str(fractions.Fraction(x).limit_denominator())})


def table(O_check, A_withB, A_m, Rate, X_Bi, C_j, Z, run_num):
    table = '<h3 id="迭代表' + str(run_num) + '"><a href="#迭代表' + str(run_num) + '" class="header-anchor">#</a> 迭代表' + str(
        run_num) + '</h2>\n<table>\n'
    # 第一行
    table += '<tr><th></th><th>C j</th>'
    for i in C_j:
        table += '<td>'
        if i == -1000000:
            table += '-M'
        else:
            table += str(Fraction(i))
        table += '</td>'
    table += '<th rowspan="2">b i</th><th rowspan="2">θ i</th></tr>\n'

    # 第二行
    table += '<tr>\n<th>C bi</th><th>X ni/j</th>'
    for i in range(1, len(C_j) + 1):
        table += '<th>'
        table += 'X' + str(i)
        table += '</th>'
    table += '</tr>\n'
    # C_bi X_bi A B rate
    for i in range(A_m):
        table += '<tr>\n'
        # C_bi
        table += '<td>'
        C_bi = C_j[int(X_Bi[i]) - 1]
        if C_bi == -1000000:
            table += '-M'
        else:
            table += str(Fraction(C_j[int(X_Bi[i]) - 1]))
        table += '</td>'
        # X_bi
        table += '<td>'
        table += 'X' + str(int(X_Bi[i]))
        table += '</td>'
        # A B
        table += '<td>'
        sio = StringIO()
        print(A_withB[i], file=sio)
        table += sio.getvalue().replace('[', '').replace(']', '').replace(' ', '</td><td>')
        table += '</td>'
        # Rate
        table += '<td>'
        if max(Rate) == 0 and min(Rate) == 0:
            pass
        else:
            Rate_list = Rate[:, 0]
            if Rate_list[i] == 1000000:
                table += '-'
            else:
                sio = StringIO()
                print(Rate_list, file=sio)
                table += sio.getvalue().replace('[', '').replace(']', '').split(' ')[i]
        table += '</td>'
        table += '</tr>\n'
    # 最后一行
    table += '<tr>\n'
    table += '<th colspan="2">σ j</th>'
    # 0_check
    for i in range(len(O_check)):
        table += '<td>'
        if O_check[i] > 10000 or O_check[i] < -10000:
            # 对2900000这样的数，取M_c为2和3,再比较M_ad两种情况下哪种比较小，取M_ad绝对值较小的那种
            M_c1 = -(-O_check[i] // 10000) / 100  # 大M的系数，为了避免负数整除会进一，比如-201/100=3
            M_c2 = (O_check[i] // 10000) / 100  # 大M的系数
            M_ad1 = O_check[i] - M_c1 * 1000000  # 大M的余项
            M_ad2 = O_check[i] - M_c2 * 1000000  # 大M的余项
            if abs(M_ad1) <= abs(M_ad2):
                M_c = M_c1
                M_ad = M_ad1
            else:
                M_c = M_c2
                M_ad = M_ad2
            # 利用np.set_printoptions可以将np数组中浮点小组转化为分数的能力,把想转化的浮点小组赋值到np数组中，再打印到StringIO，再getvalue得到string
            # print('O_check[i]\n',O_check[i],'\nM_c\n',M_c,'\nM_ad\n',M_ad)
            O_check[i] = M_c
            sio = StringIO()
            print(O_check, file=sio)
            M_c_fraction = sio.getvalue().replace('[', '').replace(']', '').split(' ')[i]  # M的系数的分数形式
            if M_c == 1 or M_c == -1:
                table += str(M_c).replace('1.0', '') + 'M'
            else:
                table += M_c_fraction + 'M'

            O_check[i] = M_ad
            sio = StringIO()
            print(O_check, file=sio)
            M_ad_fraction = sio.getvalue().replace('[', '').replace(']', '').split(' ')[i]  # M的余项的分数形式
            if M_ad == 0:
                pass
            elif M_ad > 0:
                table += '+' + M_ad_fraction
            else:
                table += M_ad_fraction
        else:
            sio = StringIO()
            print(O_check, file=sio)
            table += sio.getvalue().replace('[', '').replace(']', '').split(' ')[i]
        table += '</td>'
    # Z
    table += '<th>Z</th>'

    if Z > 10000 or Z < -10000:
        table += '<td>'
        # 对2900000这样的数，取M_c为2和3,再比较M_ad两种情况下哪种比较小，取M_ad绝对值较小的那种
        M_c1 = -(-Z // 10000) / 100  # 大M的系数，为了避免负数整除会进一，比如-201/100=3
        M_c2 = (Z // 10000) / 100  # 大M的系数
        M_ad1 = Z - M_c1 * 1000000  # 大M的余项
        M_ad2 = Z - M_c2 * 1000000  # 大M的余项
        if abs(M_ad1) <= abs(M_ad2):
            M_c = M_c1
            M_ad = M_ad1
        else:
            M_c = M_c2
            M_ad = M_ad2

        O_check[0] = M_c
        sio = StringIO()
        print(O_check, file=sio)  # 这里借用一下O_check来转Z的M_c分数
        M_c_fraction = sio.getvalue().replace('[', '').replace(']', '').split(' ')[0]  # M的系数的分数形式
        if M_c == 1 or M_c == -1:
            table += str(M_c).replace('1.0', '') + 'M'
        else:
            table += M_c_fraction + 'M'

        O_check[0] = M_ad
        sio = StringIO()
        print(O_check, file=sio)  # 这里借用一下O_check来转Z的M_ad分数
        M_ad_fraction = sio.getvalue().replace('[', '').replace(']', '').split(' ')[0]  # M的余项的分数形式
        if M_ad == 0:
            pass
        elif M_ad > 0:
            table += '+' + M_ad_fraction
        else:
            table += M_ad_fraction

        table += '</td>'

    else:
        sio = StringIO()
        O_check[0] = Z
        print(O_check, file=sio)  # 这里借用一下O_check来转分数
        table += '<td>' + sio.getvalue().replace('[', '').replace(']', '').split(' ')[0] + '</td>'
    table += '</tr>\n'

    table = table + '</table>\n<hr>\n'
    return table


def danchunrun(A, D):
    table_all = ''
    result_type = 'none'
    D1 = D.replace(' ', '').replace('\r\n', '\n').replace('f(x)', 'z').replace('F(x)', 'z').replace('F', 'z').replace(
        'Z', 'z').replace('-', '_-').replace('Min', 'mi').replace('min', 'mi').replace('MIN', 'mi').replace('Max',
                                                                                                            'ma').replace(
        'max', 'ma').replace('MAX', 'ma').replace(' ', '').replace('-x', '-1x')
    x_c_dict = {}
    if D.count('mi'):
        print('类型min')
        x_c_dict['D_type'] = 'min'
    else:
        print('类型max')
        x_c_dict['D_type'] = 'max'
    D2 = D1.replace('ma', '').replace('mi', '').replace('z=', '').replace('y=', '')
    if D2.startswith('_'):
        D2 = D2[1:]
    # print(D2)
    D3 = re.split('[+_]', D2.strip())
    print(D3)

    max_xindex = 0
    for i in D3:
        xwithindex = re.findall(r'x\d', i)[0]
        xindex = int(xwithindex.replace('x', ''))
        if xindex > max_xindex:
            max_xindex = xindex
        x_c = i.replace(xwithindex, '')
        if x_c == "":
            x_c = 1
        else:
            x_c = int(x_c)
        # print(xwithindex,x_c,xindex)
        x_c_dict[xwithindex] = x_c
    print('x有', max_xindex)

    C_j = np.zeros(max_xindex)
    for i in range(max_xindex):
        try:
            C_j[i] = x_c_dict['x' + str(i + 1)]
        except:
            x_c_dict['x' + str(i + 1)] = 0
            C_j[i] = 0

    if x_c_dict['D_type'] == 'min':
        for i in range(max_xindex):
            if C_j[i] != 0:
                C_j[i] = -C_j[i]
    # print(C_j)
    # print(x_c_dict)
    # 以上是对于目标函数D的处理，得到C_j的一部分

    # 下面开始对约束条件A做处理

    # A='x1+x2+x3<4\n2x1-x2+x3<-1\n3x2+x3=9\n'
    if A.endswith('\n') or A.endswith(';'):
        A = A[:-1]
    # print(A)
    A2 = A.replace('\r\n', '\n').replace(';', '\n').replace('\n', '\n_').replace(' ', '').replace('>=', '>').replace(
        '<=', '<').replace('-x', '-1x').replace('-', '_-').replace('__', '_').replace('>_', '>').replace('<_',
                                                                                                         '<').replace(
        '=_', '=')
    if A2.startswith('_'):
        A2 = A2[1:]
    print('A2', A2)
    x_a_dict = {}
    A3 = A2.split('\n_')
    print('A3', A3)
    A_m = len(A3)
    print('A_m', A_m)
    A = np.zeros((A_m, max_xindex))
    B = np.zeros((A_m, 1))
    for i in range(A_m):
        if '<' in A3[i]:
            A_ad = np.zeros((A_m, 1))
            A_ad[i] = 1
            A = np.concatenate([A, A_ad], axis=1)
            B_withsign = re.findall(r'<.+', A3[i])[0]
            A3[i] = A3[i].replace(B_withsign, '')
            B[i] = B_withsign.replace('<', '')
            C_j = np.append(C_j, [0])
        elif '>' in A3[i]:
            A_ad = np.zeros((A_m, 1))
            A_ad[i] = -1
            A = np.concatenate([A, A_ad], axis=1)
            B_withsign = re.findall(r'>.+', A3[i])[0]
            A3[i] = A3[i].replace(B_withsign, '')
            B[i] = B_withsign.replace('>', '')
            C_j = np.append(C_j, [0])
        elif '=' in A3[i]:
            B_withsign = re.findall(r'=.+', A3[i])[0]
            A3[i] = A3[i].replace(B_withsign, '')
            B[i] = B_withsign.replace('=', '')
            # print('B',i,B[i])
        else:
            print('出错<>=都没找到')
        # 以上把1和-1的标准型加入了A，读取了B,但约束条件系数未 读取
        # print('A3',A3)
        A4 = re.split('[+_]', A3[i].strip())
        print(i+1,'行',A4)
        for j in A4:
            x_withindex = re.findall(r'x\d', j)[0]
            x_index = int(x_withindex.replace('x', ''))
            x_c = j.replace(x_withindex, '')
            if x_c == '':
                x_c = 1
            print("A",A,"i",i,"x_index-1",x_index-1)
            A[i][x_index - 1] = int(x_c)
    # 下面检查有没有B<0
    for i in range(A_m):
        if B[i] < 0:
            B[i] = -B[i]
            A[i] = -A[i]
    # print('A',A)
    # print('B',B)
    # print('C_j',C_j)

    # 上面进行了简单的化标准型处理，得到了A,B,但未曾添加人工变量
    # 下面将判断基变量是否够，若不够，将添加人工变量
    X_Bi = np.zeros(A_m)
    X_manual = []
    # print('A\n',A)
    for i in range(A_m):
        for j in range(len(C_j)):
            if j + 1 in X_Bi:
                continue
            if A[i][j] == 1:
                import copy
                A_check_n = copy.deepcopy(A[:, j])
                # print('A3\n',i,j,A)
                A_check_n[i] = 0
                # print(A_check_n)
                # print('A1\n',i,j,A)
                if max(A_check_n) == 0 and min(A_check_n) == 0:
                    print('第', i + 1, '行有基变量 x', j + 1)
                    X_Bi[i] = j + 1
                    # print('A2\n',A)
        if X_Bi[i] != 0:
            continue
        A_ad = np.zeros((A_m, 1))
        A_ad[i] = 1
        A = np.concatenate([A, A_ad], axis=1)
        C_j = np.append(C_j, [-1000000])
        X_Bi[i] = len(C_j)
        X_manual = np.append(X_manual, [len(C_j)])
    A_withB = np.concatenate([A, B], axis=1)
    print('A_withB', A_withB)  # 存储系数增广矩阵
    print('C_j', C_j)  # 存储所有X的系数
    print('X_Bi', X_Bi)  # 存储基变量的X序号
    print('X_manual', X_manual)
    # 上面得出了最终的初始迭代表

    run_num = 0
    # 迭代循环部分
    # 开始迭代输入A_withB,X_Bi,C_j
    while (result_type == 'none'):
        run_num += 1
        print('\n_______________________\n')
        print('************第', run_num, '次')
        print('_______________________\n')
        print('A_withB:\n', A_withB)
        print('X_Bi\n', X_Bi)
        A = np.delete(A_withB, -1, axis=1)
        # print('A:',A)
        B = A_withB[:, -1]
        # print('B:',B)
        C_Bi = np.zeros(A_m)

        for i in range(A_m):
            C_Bi[i] = C_j[int(X_Bi[i] - 1)]
        print('C_Bi\n', C_Bi)  # 存储基变量系数

        O_check = copy.deepcopy(C_j)
        for i in range(A_m):
            O_check = O_check - C_Bi[i] * A[i]
        print('O_check\n', O_check)  # 存储检验数

        if max(O_check) > 0:
            print('有非负检验数')
            # O_check_maxindex=np.argsort(-O_check)[0]

            O_check_maxindex = np.argwhere(O_check == max(O_check))[0]
            X_Bi_in = O_check_maxindex + 1
            print('X_Bi_in', X_Bi_in)  # 确认换入变量
            # 后判断是否有换出变量符合条件
            exist_O_check = 'is'
        else:
            print('已无非负检验数')
            # 后跟判断解的类型
            exist_O_check = 'not'

        if exist_O_check == 'is':  # 1.判断是否有换出变量符合条件
            Rate = np.zeros((A_m, 1))
            for i in range(A_m):
                if A[i][O_check_maxindex] <= 0 or B[i] <= 0:
                    Rate[i] = 1000000
                else:
                    Rate[i] = B[i] / A[i][O_check_maxindex]
            # print('Rate:',Rate)

            Z = 0
            for i in range(A_m):
                C_bi = C_j[int(X_Bi[i]) - 1]
                # print('准备为Z',B)
                B_i = B[i]
                Z += C_bi * B_i

            table_all += table(O_check, A_withB, A_m, Rate, X_Bi, C_j, Z, run_num)

            # 下面开始找换出基变量
            if min(Rate) == 1000000:
                print('无界解')
                result_type = '无界解'
            else:
                Rate_list = Rate[:, 0]
                print('Rate_list', Rate_list)
                Rate_min_index = np.argwhere(Rate_list == min(Rate_list))[:, 0]
                if len(Rate_min_index) > 1:  # 如果有两个或以上相同最小比值
                    Rate_min_X_Bi = X_Bi[Rate_min_index]
                    Rate_min_index = np.argwhere(Rate_min_X_Bi == max(Rate_min_X_Bi))[0]
                    X_Bi_out = Rate_min_X_Bi[Rate_min_index]
                else:  # 如果最小比值只有一个
                    Rate_min_index = Rate_min_index[0]
                    X_Bi_out = X_Bi[Rate_min_index]
                print('X_Bi_out', X_Bi_out)
                Rate_min_index = np.argwhere(X_Bi == X_Bi_out)[0]
                X_Bi[Rate_min_index] = X_Bi_in  # 基变量换入换出
                # print('基变量变换后',X_Bi)
                A_withB[Rate_min_index] = A_withB[Rate_min_index] / A_withB[Rate_min_index, X_Bi_in - 1][0]
                # print(A_withB)
                for i in range(A_m):
                    if i != Rate_min_index:
                        A_withB[i] = A_withB[i] - A_withB[i][X_Bi_in - 1] * A_withB[Rate_min_index]
                # print('A_withB\n',A_withB)
                # print('X_Bi\n',X_Bi)
        elif exist_O_check == 'not':  # 后跟判断解的类型
            Rate = np.zeros((A_m, 1))
            Z = 0
            for i in range(A_m):
                C_bi = C_j[int(X_Bi[i]) - 1]
                # print('准备为Z',B)
                B_i = B[i]
                Z += C_bi * B_i
            table_all += table(O_check, A_withB, A_m, Rate, X_Bi, C_j, Z, run_num)

            for i in X_Bi:
                if i in X_manual:
                    print('最终解中存在人工变量,无可行解')
                    result_type = '无可行解'
                    break
            if result_type == 'none':
                for i in range(len(C_j)):
                    # print('检查X',i+1)
                    if i + 1 not in X_Bi:
                        if O_check[i] == 0:
                            result_type = '无穷多最优解'
                            break
            if result_type == 'none':
                result_type = '唯一最优解'
                # print('A_withB\n',A_withB)
                # print('X_Bi\n',X_Bi)
        else:
            print('出错')

    print('result_type', result_type)
    # print(table_all)
    return table_all, result_type

if __name__=="__main__":
    A='x1-2x2-x3=-2\n-x1-x2-x4=-4\n-2x1+x2-x5=-5'
    D='max z=x1+2x2+0x3+0x4+0x5'
    res, _=danchunrun(A,D)
    print(res)
