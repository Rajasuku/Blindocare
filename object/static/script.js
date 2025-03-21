function updateObjects() {
    fetch('/get_objects')
        .then(response => response.json())
        .then(data => {
            let objectsList = document.getElementById("objects_list");
            objectsList.innerHTML = "";

            data.forEach(obj => {
                let listItem = document.createElement("li");
                listItem.textContent = `${obj[0]} - ${obj[1]}m (${obj[2]})`;
                objectsList.appendChild(listItem);

                // Play Audio Feedback
                let speech = new SpeechSynthesisUtterance(`${obj[0]} is ${obj[1]} meters on ${obj[2]}`);
                speech.rate = 1;
                window.speechSynthesis.speak(speech);
            });
        });
}

// Update detected objects every 3 seconds
setInterval(updateObjects, 3000);
