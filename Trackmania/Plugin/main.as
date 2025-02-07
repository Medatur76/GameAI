vec3 lastPos = vec3(0.0, 0.0, 0.0);

bool running = false;

void respond(Net::Socket@ conn)
{
    PlayerState::sTMData@ TMData = PlayerState::GetRaceData();
    auto PlayerInfo = TMData.dPlayerInfo;
    float distance = 0.0;
    if (lastPos == vec3(0.0, 0.0, 0.0)) distance = Math::Sqrt((PlayerInfo.Position.x)**2+(PlayerInfo.Position.y)**2);
    else distance = Math::Sqrt((PlayerInfo.Position.x-lastPos.x)**2+(PlayerInfo.Position.y-lastPos.y)**2);
    lastPos = PlayerInfo.Position;
    conn.Write("{ \"running\": \"" + running + "\", \"speed\": " + PlayerInfo.Speed + ", \"position\": [" + PlayerInfo.Position.x + ", " + PlayerInfo.Position.y + ", " + PlayerInfo.Position.z + "], \"end\": \"" + TMData.dEventInfo.EndRun + "\" }");
}

void OnKeyPress(bool down, VirtualKey key) {
    if (down && key == VirtualKey::J) {
        running = !running;
        if (running) print("AI Started!");
        else print("AI Stoped!");
    }
}

void Main() {
    auto server = Net::Socket();
    while (!server.Listen("127.0.0.1", 9000)) {
        yield();
    }
    while(!server.IsReady()){
        yield();
    }
    print("Server started!\r\n");

    while(true) {
        auto conn = server.Accept(); // 1000 ms timeout
        if (conn is null) {
            yield();
            continue;
        }
        while (!conn.IsReady()) {
            yield();
        }
        respond(conn);
        conn.Close();
    }
}
