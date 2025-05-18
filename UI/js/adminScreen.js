var options = {};
var elems = {};
var fs = require('fs');
var path = require('path');
var config = {
    apiKey: "AIzaSyAlkTeBTBPtnVAaOnNmwiwsFVDIWKhfp5M",
    authDomain: "hackathon-mozofest-2019.firebaseapp.com",
    databaseURL: "https://hackathon-mozofest-2019.firebaseio.com",
    storageBucket: 'gs://hackathon-mozofest-2019.appspot.com/'
};

firebase.initializeApp(config);
var user = firebase.auth().currentUser;

firebase.auth().onAuthStateChanged(function(user) {
    if (user) {
        console.log("Loggd in already : " + user.email);
        M.toast({html:'Welcome back ' + user.email + ' !'});
    }
  });

function loadAttendance(day) {
    const csvPath = path.join(__dirname, '..', 'Attendance.csv');
    fs.readFile(csvPath, 'utf8', (err, data) => {
        if (err) {
            console.error('Error reading attendance:', err);
            document.getElementById('attendanceTable').innerHTML = '<tr><td colspan="2">Could not load attendance.</td></tr>';
            return;
        }
        const lines = data.trim().split('\n');
        const headers = lines[0].split(',');
        const dayIndex = headers.indexOf(day);
        let html = '<table class="striped centered"><thead><tr><th>Reg No</th><th>Present</th></tr></thead><tbody>';
        for (let i = 1; i < lines.length; i++) {
            const cols = lines[i].split(',');
            html += `<tr><td>${cols[0]}</td><td>${cols[dayIndex]}</td></tr>`;
        }
        html += '</tbody></table>';
        document.getElementById('attendanceTable').innerHTML = html;
    });
}

document.addEventListener('DOMContentLoaded', function() {
    elems = document.querySelectorAll('select');
    M.FormSelect.init(elems, options);
    // Initial load for Day1
    loadAttendance('Day1');
    document.querySelector('select').addEventListener('change', function() {
        const day = 'Day' + this.value;
        loadAttendance(day);
    });
});

$('select').on('change', function() {
    // console.log($(this).val());
    foo($(this).val());
});
var selection;
function foo(selection){
    console.log(selection + " from inside foo along with user : " + user);
        switch(selection){
            case '1': document.location.href = "etc/data1.html";
                    break;
            case '2': document.location.href = "etc/data2.html";
                    break;
            case '3': document.location.href = "etc/data3.html";
                    break;
        }

        //fetch from database
        // Admin signed in.
      
}

function logOut(){
	console.log("Attempting Sign Out");
	firebase.auth().signOut().then(function() {
    	console.log("Sign out successful");
    	document.location.href = "adminLogin.html";
  	}).catch(function(error) {
	    console.log("Error singing out");
  	});
}
function pyCam(){
    console.log("Camera Input Running");
    M.toast({html:'Opening Camera Feed'});
    var python = require('child_process').spawn('python', ['py/camcap2.py']);
    python.stdout.on('data',function(data){
        console.log("data: ",data.toString('utf8')+ " from Python ");
        // Optionally, refresh attendance display here
    });
}

function sendCSV(){
    

}

console.log("JS ready");