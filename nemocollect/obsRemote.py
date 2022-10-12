#!python3
import sys
import asyncio

from obswsrc import OBSWS
from obswsrc.requests import ResponseStatus, StartRecordingRequest, StopRecordingRequest
from obswsrc.types import Stream, StreamSettings

async def main():
    if len(sys.argv) < 2:
        print("usage: python3.6 obsRemote.py <start/stop>")
        return -1
    
    async with OBSWS('localhost', 4444, "password") as obsws:
        # Now let's actually perform a request
        response = await obsws.require(StartRecordingRequest() if sys.argv[1] == "start" else StopRecordingRequest())

        # Check if everything is OK
        if response.status == ResponseStatus.OK:
            print("OK")
        else:
            print("ERROR: ", response.error)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
