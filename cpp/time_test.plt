set terminal svg
set output "time_test.svg"
set ytics autofreq
set xtics autofreq
set border 3
set bar 0.5
set xtics nomirror
set ytics nomirror
set xrange [*:*]
set yrange [*:*]
set grid
#set xtics 0.5
#set ytics 0.5
#set xzeroaxis lt -1
#set yzeroaxis lt -1
#set xtics axis
#set ytics axis
#set key box
#название таблицы
#set title "Зависимость квадрата радиуса кольца от номера кольца"
#set key outside bottom
#название осей
set ylabel "time, s"
set xlabel "n"
f(x) =  A * exp(b*x + C)
A = 1
b = 1
C = 1
fit f(x) "base_result.csv" u 1:5 via A, b, C
print(f(22))
set logscale y
plot 'base_result.csv' u 1:5 pt 5 ps 0.5 notitle, f(x) notitle
set table "data.dat"
replot
unset table
