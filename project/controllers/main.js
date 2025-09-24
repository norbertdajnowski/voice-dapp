//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording
const recordings = [];


// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record

var recordButton = document.getElementById("recordButton");

var recordLoginButton = document.getElementById("recordLoginButton");

//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);


function register(){
    var usernameRegister = document.getElementById("usernameRegister").value;
   
    if (recordings.length == 3 && usernameRegister != ""){

        var form = new FormData();

        const tempBlob = new Blob([usernameRegister], {type : 'text/plain'})

        form.append('voice1', recordings[0]);
        form.append('voice2', recordings[1]);
        form.append('voice3', recordings[2]);
        form.append('username', tempBlob);

        $.ajax({
            type: 'POST',
            url: '/addVoice',
            data: form,
            contentType: false,
            processData: false
        }).done(function(data) {
            alert("Registration succesful");
            window.location.replace("http://");
        });

    }else{
        return;
    }
}

function login(){
    var usernameRegister = document.getElementById("usernameLogin").value;
   
    if (recordings.length == 1 && usernameLogin != ""){
        var form = new FormData();

        const tempBlob = new Blob([usernameRegister], {type : 'text/plain'})

        form.append('voice1', recordings[0]);
        form.append('username', tempBlob);

        $.ajax({
            type: 'POST',
            url: '/recognise',
            data: form,
            contentType: false,
            processData: false
        }).done(function(data) {
            alert(data);
        });
    }
}

function test(){

        $.ajax({
            type: 'POST',
            url: '/test',
            data: form,
            contentType: false,
            processData: false
        }).done(function(data) {
            alert(data);
        });

}

function del(){
    var usernameDelete = document.getElementById("usernameDelete").value;
    var form = new FormData();

    const tempBlob = new Blob([usernameDelete], {type : 'text/plain'})

    form.append('username', tempBlob);
    
    $.ajax({
        type: 'POST',
        url: '/deleteVoice',
        data: form,
        contentType: false,
        processData: false
    }).done(function(data) {
        alert(data);
    });
}


function startRecording() {
    if (document.getElementById("recordingsList").getElementsByTagName("li").length < 3){

        /*
            Simple constraints object, for more advanced audio features see
            https://addpipe.com/blog/audio-constraints-getusermedia/
        */
        
        var constraints = { audio: true, video:false }

        /*
            Disable the record button until we get a success or fail from getUserMedia() 
        */

        recordButton.disabled = true;

        /*
            We're using the standard promise based getUserMedia() 
            https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
        */

        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

            /*
                create an audio context after getUserMedia is called
                sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
                the sampleRate defaults to the one set in your OS for your playback device
            */
            audioContext = new AudioContext();

            //update the format 
            document.getElementById("formats").innerHTML="Format: 2 channel pcm @ "+audioContext.sampleRate+"Hz"
            /*  assign to gumStream for later use  */
            gumStream = stream;
            
            /* use the stream */
            input = audioContext.createMediaStreamSource(stream);

            /* 
                Create the Recorder object and configure to record mono sound (1 channel)
                Recording 2 channels  will double the file size
            */
            rec = new Recorder(input,{numChannels:2})

            //start the recording process
            rec.record()

            console.log("Recording started");

            setTimeout(function() {
                    //tell the recorder to stop the recording
                rec.stop();
                
                recordButton.disabled = false;

                //stop microphone access
                gumStream.getAudioTracks()[0].stop();

                //create the wav blob and pass it on to createDownloadLink
                rec.exportWAV(createDownloadLink);
            }, 3000);

        }).catch(function(err) {
            //enable the record button if getUserMedia() fails
            alert("Error encountered");
            recordButton.disabled = false;
        });
    }else{
        alert("You have completed all the recordings, please proceed with the form.")
    }

}

function startLoginRecording() {
    if (document.getElementById("recordingsList").getElementsByTagName("li").length < 1){

        /*
            Simple constraints object, for more advanced audio features see
            https://addpipe.com/blog/audio-constraints-getusermedia/
        */
        
        var constraints = { audio: true, video:false }

        /*
            Disable the record button until we get a success or fail from getUserMedia() 
        */

        recordLoginButton.disabled = true;

        /*
            We're using the standard promise based getUserMedia() 
            https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
        */

        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

            /*
                create an audio context after getUserMedia is called
                sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
                the sampleRate defaults to the one set in your OS for your playback device
            */
            audioContext = new AudioContext();

            //update the format 
            document.getElementById("formats").innerHTML="Format: 2 channel pcm @ "+audioContext.sampleRate+"Hz"
            /*  assign to gumStream for later use  */
            gumStream = stream;
            
            /* use the stream */
            input = audioContext.createMediaStreamSource(stream);

            /* 
                Create the Recorder object and configure to record mono sound (1 channel)
                Recording 2 channels  will double the file size
            */
            rec = new Recorder(input,{numChannels:2})

            //start the recording process
            rec.record()

            console.log("Recording started");

            setTimeout(function() {
                    //tell the recorder to stop the recording
                rec.stop();
                
                recordLoginButton.disabled = false;

                //stop microphone access
                gumStream.getAudioTracks()[0].stop();

                //create the wav blob and pass it on to createDownloadLink
                rec.exportWAV(createDownloadLink);
            }, 3000);

        }).catch(function(err) {
            //enable the record button if getUserMedia() fails
            alert("Error encountered");
            recordLoginButton.disabled = false;
        });
    }else{
        alert("You have completed all the recordings, please proceed with the form.")
    }

}

function createDownloadLink(blob) {
	
	var url = URL.createObjectURL(blob);
    recordings.push(blob);
	var au = document.createElement('audio');
	var li = document.createElement('li');
	var link = document.createElement('a');

	//name of .wav file to use during upload and download (without extendion)
	var filename = new Date().toISOString();

	//add controls to the <audio> element
	au.controls = true;
	au.src = url;

	//save to disk link
	link.href = url;
	link.download = filename+".wav"; //download forces the browser to donwload the file using the  filename
	link.innerHTML = "Save to disk";

	//add the new audio element to li
	li.appendChild(au);
	
	//add the filename to the li
	li.appendChild(document.createTextNode(filename+".wav "))

	//add the save to disk link to li
	li.appendChild(link);

	//add the li element to the ol
	recordingsList.appendChild(li);
}