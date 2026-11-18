[ ! -d "./bin" ] && mkdir "./bin"

g++ -I./include main.cpp src/.*\.(c$)|(cpp$) -o ./bin/program

chmod +x ./bin/program

./bin/program

echo $?