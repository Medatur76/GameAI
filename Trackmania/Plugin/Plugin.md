# Plugin

This is where all the Angel Script for this AI can be found. It doesnt look like much but the data that this code provides is crucial for the AI to function. Lets break it down into 2 sections: [Server Setup](#server-setup) and [Sending Data](#responding).

## Server Setup

```
void Main() {
    auto server = Net::Socket();
    while (!server.Listen("127.0.0.1", 9000)) {
        yield();
    }
    while(!server.IsReady()){
        yield();
    }
    print("Server started!\r\n");
    ...
}
```

This is the part of the plugin where we setup the server ``127.0.0.1:9000`` so we can recieve data from the game that will be used to train the AI.  
We begin in the ``Main()`` function by defining the server as a ``Net::Socket`` so we can send a recieve data. We then tell it to listen to ``127.0.0.1:9000``. If it fails to do so we wait a frame then try again until it does. We then wait until ites ready, checking every frame. Now a server is setup! We can now move on to [Sending Data](#responding).

## Responding

```

void respond(Net::Socket@ conn)
{
    PlayerState::sTMData@ TMData = PlayerState::GetRaceData();
    conn.Write("{ \"speed\": " + TMData.dPlayerInfo.Speed + ", \"status\": \"success\" }");
}

void Main() {
  ...

  while(true) {
        auto conn = server.Accept();
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
```

This is the part of the code where we accept a connection and send it player data.  
We begin again in the function ``Main()`` where we start a loop that runs forever. The server tries to accept a connection and same it as ``conn`` If there isnt a connection (aka ``conn`` is null) then we wait a frame then restart the loop. When there is a connection we wait until its ready, send it data, and close the connection.  
  
The data sending part is found in the function ``respond(Net::Socket@)``. We begin by using the [PlayerState Info](https://openplanet.dev/plugin/playerstate) plugin to gain easy data about the race. This data is then passed into a string resembling a JSON file which is send to the socket that was passed.
