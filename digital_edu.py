import pandas as pd

df = pd.read_csv('train.csv')
#print(df.head(2000))



df.drop(['relation', 'people_main', 'city', 
'occupation_name', 'langs', 'bdate', 'graduation', 
'last_seen', 'education_status', 'followers_count', 'life_main','id' ], axis = 1, inplace = True)

man = 0 
woman = 0

def mw_bought(row):
    global man, woman
    if row['sex'] == 2 and row['result'] == 1:
        man += 1
    if row['sex'] == 1 and row['result'] == 1:
        woman += 1
    return False

df.apply(mw_bought, axis = 1)
s = pd.Series(data= [man, woman],
index = ['ЖЕНЩИНЫ', 'МУЖЧИНЫ'])

s.plot(kind = 'barh')
plt.show()

def fill_educat(education_form):
    if education_form == 'Distance Learning':
        return 1
    return 2
df['education_form'] = df['education_form'].apply(fill_educat)

def fill_octp(occupation_type):
    if occupation_type == 'work':
        return 1
    return 2
df['occupation_type'] = df['occupation_type'].apply(fill_octp)

def fill_cs(career_start):
    if career_start == 'False':
        return 1
    return 2
df['career_start'] = df['career_start'].apply(fill_cs)

def fill_csend(career_end):
    if career_end == 'False':
        return 1
    return 2
df['career_end'] = df['career_end'].apply(fill_csend)

def fill_phon(has_mobile):
    if has_mobile == 1.0:
        return 1
    return 2
df['has_mobile'] = df['has_mobile'].apply(fill_phon)

df.info()


print(df.head(50))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

x = df.drop('result', axis = 1)
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(y_test)
print(y_pred)
print(round(accuracy_score(y_test, y_pred) * 100, 2))
print('aaaa')
print(confusion_matrix(y_test, y_pred))






