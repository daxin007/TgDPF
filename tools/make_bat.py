import itertools
import datetime

start = 2800
prefix = f'js_start_date_{start}'
working_dir = r'C:\Users\Administrator\Desktop\DT'
 
start_date = [ 
    '2020-07-01',# 4 months
    '2020-06-01',# 5 months
    '2020-05-01',# 6 months
    '2020-04-01',# 7 months,
    
    '2020-03-01',# 8 months
    '2020-02-01',# 9 months
    '2020-01-01',# 10 months
]
kfold_start_date = [datetime.date(2020, m, 1).strftime('%Y-%m-%d') for m in (1,3,5,7,9,11)]
noisy_ratio = [0.05, 0.1, 0.2, 0.4, 0.6]

# seed = [666666, 555, 3223, 5151, 7777, 90234]
# seed = [3333333, 5555555, 4234124, 7577456, 42412, 55]
# seed = [666666, 555, 3223, 5151, 7777, 3333333, 5555555, 4234124, 7577456, 42412]
seed = [0, 1111, 2222, 3333, 4444]
#parameters = [start_date]
parameters = [kfold_start_date]


for i, parameter in enumerate(list(itertools.product(*parameters))):
    for j in range(5):
        # 'cmd /k "cd /d C:\Users\Admin\Desktop\venv\Scripts & activate & cd /d    C:\Users\Admin\Desktop\helloworld & python manage.py runserver"'
        # cmd = f'C:/Users/Administrator/.venv/machine_learning/Scripts/python.exe c:/Users/Administrator/Desktop/大唐风电/data_dayahead.py --id {start} --seed {seed[j]} -- start {parameter[0]}'
        bat_filename = f'run/{prefix}_{i}_{start//5}.bat'
        with open(bat_filename, 'a', encoding='utf-8') as f:
            f.write('echo Starting...\n')
            # f.write(r'.venv\machine_learning\Scripts\activate')
            # f.write('\n')
            # f.write('cd {}\n'.format(working_dir))
            f.write("start cmd /k \"cd /d C:/Users/Administrator/.venv/machine_learning/Scripts & activate & cd /d C:/Users/Administrator/Desktop/DT & python train_dayahead.py --id {} --seed {} --start {} --js_mode --js_only\"\n\n".format(start, seed[j], parameter[0]))
        start += 1