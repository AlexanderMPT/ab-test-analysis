import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom_test, ttest_ind
import os

# ---------- Настройка ----------
DATA_FILE = 'hw_ab.csv'          # Измените, если файл называется vis.csv
OUTPUT_FILE = 'output.txt'       # Текстовые вывод
PLOTS_DIR = 'plots'              # Папка для графиков

# Создаём папку для графиков, если её нет
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- Загрузка данных ----------
df = pd.read_csv(DATA_FILE)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(f'Размер данных: {df.shape}\n\n')

# ---------- Проверка структуры ----------
with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
    f.write('Проверка пропусков:\n')
    f.write(str(df.isnull().sum()) + '\n\n')
    f.write('Уникальные значения по колонкам:\n')
    for col in df.columns:
        f.write(f'{col}: {df[col].nunique()}\n')

# ---------- Уникальность пользователей ----------
dupl = df['id'].duplicated().sum()
with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
    f.write(f'\nДубликатов id: {dupl}\n')
    f.write(f'Всего строк: {len(df)}, уникальных id: {df["id"].nunique()}\n')

# ---------- Балансировка групп ----------
group_counts = df['group'].value_counts()
daily_stats = df.groupby(['date','group']).agg(users=('id','count'), conversions=('converted','sum'))
pivot = daily_stats['users'].unstack(fill_value=0)

with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
    f.write('\nРаспределение по группам:\n')
    f.write(str(group_counts) + '\n')
    if (pivot == 0).any().any():
        f.write('⚠️ Есть дни без посетителей в одной из групп!\n')
    else:
        f.write('Каждый день в обеих группах есть посетители.\n')

# ---------- Общие конверсии ----------
control = df[df['group'] == 'control']
test = df[df['group'] == 'test']

conv_control = control['converted'].sum()
seen_control = len(control)
conv_test = test['converted'].sum()
seen_test = len(test)

rate_control = conv_control / seen_control
rate_test = conv_test / seen_test
lift = (rate_test / rate_control) - 1

with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
    f.write(f'\nКонверсия контроль: {conv_control}/{seen_control} = {rate_control:.4f} ({rate_control*100:.2f}%)\n')
    f.write(f'Конверсия тест:     {conv_test}/{seen_test} = {rate_test:.4f} ({rate_test*100:.2f}%)\n')
    f.write(f'Относительный прирост: {lift*100:.2f}%\n')

# ---------- Биномиальный тест ----------
p_binom = binom_test(conv_test, n=seen_test, p=rate_control, alternative='two-sided')
alpha = 0.05
with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
    f.write(f'\nБиномиальный тест: p-value = {p_binom:.4f}\n')
    f.write('Значимо на уровне 0.05\n' if p_binom < alpha else 'Не значимо на уровне 0.05\n')

# ---------- Длительность теста ----------
unique_dates = sorted(df['date'].unique())
dates_series = pd.to_datetime(unique_dates)
date_diff = dates_series[1:] - dates_series[:-1]
all_consecutive = all(date_diff == pd.Timedelta(days=1))

with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
    f.write(f'\nТест длился {len(unique_dates)} дней ({unique_dates[0]} – {unique_dates[-1]})\n')
    f.write('Пропусков дат нет.\n' if all_consecutive else 'Обнаружены пропуски дат.\n')

# ---------- Дневные метрики ----------
test_n = test.groupby('date')['converted'].sum()
control_n = control.groupby('date')['converted'].sum()
common_dates = test_n.index.intersection(control_n.index)
test_n = test_n.loc[common_dates]
control_n = control_n.loc[common_dates]

test_users = test.groupby('date')['id'].count()
control_users = control.groupby('date')['id'].count()
test_conv_rate = test_n / test_users.loc[common_dates]
control_conv_rate = control_n / control_users.loc[common_dates]

# ---------- Визуализации ----------
# 1. Количество оформленных карт по дням
plt.figure(figsize=(12, 5))
sns.lineplot(x=test_n.index, y=test_n.values, color='red', label='Test')
sns.lineplot(x=control_n.index, y=control_n.values, color='blue', label='Control')
plt.title('Количество оформленных карт по дням')
plt.xlabel('Дата'); plt.ylabel('Конверсии')
plt.ylim(0, None)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'daily_conversions.png'))
plt.close()

# 2. Конверсия по дням
plt.figure(figsize=(12, 5))
sns.lineplot(x=test_conv_rate.index, y=test_conv_rate.values, color='red', label='Test')
sns.lineplot(x=control_conv_rate.index, y=control_conv_rate.values, color='blue', label='Control')
plt.title('Дневная конверсия')
plt.xlabel('Дата'); plt.ylabel('Конверсия')
plt.ylim(0, None)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'daily_conversion_rate.png'))
plt.close()

# 3. Гистограммы дневной конверсии
plt.figure(figsize=(10, 5))
plt.hist(test_conv_rate, bins=20, density=True, color='red', alpha=0.3, label='Test')
plt.hist(control_conv_rate, bins=20, density=True, color='blue', alpha=0.3, label='Control')
plt.title('Распределение дневной конверсии')
plt.xlabel('Конверсия'); plt.ylabel('Плотность')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'hist_conversion_rate.png'))
plt.close()

# 4. Гистограммы дневного количества конверсий
plt.figure(figsize=(10, 5))
plt.hist(test_n, bins=20, density=True, color='red', alpha=0.3, label='Test')
plt.hist(control_n, bins=20, density=True, color='blue', alpha=0.3, label='Control')
plt.title('Распределение дневного числа конверсий')
plt.xlabel('Число конверсий в день'); plt.ylabel('Плотность')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'hist_daily_conversions.png'))
plt.close()

# ---------- t-тесты ----------
t_conv, p_conv = ttest_ind(test_conv_rate, control_conv_rate, equal_var=False)
t_n, p_n = ttest_ind(test_n, control_n, equal_var=False)

with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
    f.write(f'\n--- t-тест для дневной конверсии ---\n')
    f.write(f't = {t_conv:.4f}, p-value = {p_conv:.4f}\n')
    f.write('Значимое различие\n' if p_conv < alpha else 'Нет значимого различия\n')
    f.write(f'\n--- t-тест для дневного числа конверсий ---\n')
    f.write(f't = {t_n:.4f}, p-value = {p_n:.4f}\n')
    f.write('Значимое различие\n' if p_n < alpha else 'Нет значимого различия\n')

# ---------- Финальное сообщение ----------
with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
    f.write('\nАнализ завершён. Графики сохранены в папке plots/\n')
