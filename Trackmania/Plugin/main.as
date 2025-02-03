vec3 lastPos = null;

bool running = false;

void respond(Net::Socket@ conn)
{
    PlayerState::sTMData@ TMData = PlayerState::GetRaceData();
    auto PlayerInfo = TMData.dPlayerInfo;
    float distance = 0.0;
    if (lastPos is null) distance = Math.sqrt((PlayerInfo.Position.x)**2+(PlayerInfo.Position.y)**2);
    else distance = Math.sqrt((PlayerInfo.Position.x-lastPos.x)**2+(PlayerInfo.Position.y-lastPos.y)**2);
    lastPos = PlayerInfo.Position;
    conn.Write("{ \"running\": \"" + running + "\", \"speed\": " + PlayerInfo.Speed + ", \"last_distanced_traveled\": \"" + distance + "\", \"end\": \"" + TMData.dEventInfo.EndRun + "\" }");
}

void onKeyPress(bool down, VirtualKey key) {
    if (down && key == VirtualKey::J) {
        running = !running;
        if (brunning) print("AI Started!");
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
