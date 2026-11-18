@ECHO OFF

if not exist "./bin" mkdir "./bin"

g++ -o ./bin/program -I./include main.cpp src/*.c src/*.cpp

"./bin/program.exe"

echo %ERRORLEVEL%