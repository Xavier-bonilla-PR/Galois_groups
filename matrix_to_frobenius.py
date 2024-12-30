#Works but not for polynomials of degree 10 or higher

import subprocess


def is_prime(number):
    if number > 1:
        for num in range(2, int(number**0.5) + 1):
            if number % num == 0:
                return False
        return True
    return False

def run_pari(command):
    result = subprocess.run(['gp', '-q'], input=command, text=True, capture_output=True)
    return result.stdout.strip()

# Example usage
pari_matrix = input("matrix: ")


x = run_pari(f"charpoly({pari_matrix})")
print(x)

factormodlist = []
for num in range(20):

    if is_prime(num):
        y = run_pari(f"factormod({x}, {num})")
        factormodlist.append(y)

i = 0
for mod in factormodlist:


    t = factormodlist[i].replace(" ", "").replace("\n", "")
    e = t.split("][")

    frobenius = []
    for item in e:
        #degree
        s = item.split("x")
        #print(s)
        r = s[1]     

        if str(r)[0] == '^':
            degree = str(r)[1]
        else:
            degree = '1'
        #print(f'degree: {degree}')

        #repeats
        if item[-1] == ']':
            repeat_num = item[-2]
            #print(f'repeat: {repeat_num}')
        else:
            repeat_num = item[-1]
            #print(f'repeat: {repeat_num}')
        
        #builds frobenius element for 
        
        for v in range(int(repeat_num)):
            frobenius.append(degree)
    print(frobenius)

    i = i + 1
    

#[-1, 1, 1; -1, -1, 1; -1, -1, -1]   
#[-2, 0, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 1, 1, 1, 1]
#[0, 0, 0, 0, -1; 1, 0, 0, 0, -1; 0, 1, 0, 0, -1; 0, 0, 1, 0, -1; 0, 0, 0, 1, -1]
#[0, 0, 0, 0, 0, 0, 0, 1; 1, 0, 0, 0, 0, 0, 0, 0; 0, 1, 0, 0, 0, 0, 0, 0; 0, 0, 1, 0, 0, 0, 0, 0; 0, 0, 0, 1, 0, 0, 0, 0; 0, 0, 0, 0, 1, 0, 0, 0; 0, 0, 0, 0, 0, 1, 0, 0; 0, 0, 0, 0, 0, 0, 1, 0]
#[0, 0, 0, 0, 0, 0, 0, 0, 0, -1; 1, 0, 0, 0, 0, 0, 0, 0, 0, -1; 0, 1, 0, 0, 0, 0, 0, 0, 0, -1; 0, 0, 1, 0, 0, 0, 0, 0, 0, -1; 0, 0, 0, 1, 0, 0, 0, 0, 0, -1; 0, 0, 0, 0, 1, 0, 0, 0, 0, -1; 0, 0, 0, 0, 0, 1, 0, 0, 0, -1; 0, 0, 0, 0, 0, 0, 1, 0, 0, -1; 0, 0, 0, 0, 0, 0, 0, 1, 0, -1; 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]
