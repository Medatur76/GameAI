void respond(Net::Socket@ conn)
{
    PlayerState::sTMData@ TMData = PlayerState::GetRaceData();
    auto PlayerInfo = TMData.dPlayerInfo;
    conn.Write("{ \"status\": \"success\", \"speed\": " + PlayerInfo.Speed + ", \"position\": { \"x\": \"" + PlayerInfo.Position.x + "\", \"y\": \"" + PlayerInfo.Position.y + "\", \"z\": \"" +  PlayerInfo.Position.z + "\" } }");
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
