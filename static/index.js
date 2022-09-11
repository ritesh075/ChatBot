let name = "You";
let textarea = document.querySelector('#textarea')
let messageArea = document.querySelector('.message__area')
// do {
//     name = prompt('Please enter your name: ')
// } while(!name)

textarea.addEventListener('keyup', (e) => {
    if(e.key === 'Enter') {
        sendMessage(e.target.value)
    }
})

function sendMessage(message) {
    let msg = {
        user: name,
        message: message.trim()
    }
    // Append 
    appendMessage(msg, 'outgoing')
    textarea.value = ''
    scrollToBottom()

    // Send to server 
    // socket.emit('message', msg)
    const data = { q: message.trim() };
    const apiUrl = 'https://chanshu19.pythonanywhere.com/api/respond'
    fetch(apiUrl, {
    method: 'POST', // or 'PUT'
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
    })
    .then((response) => response.json())
    .then((data) => {
        console.log(data['answer'])
        let msg = {
            user: "Bot",
            message: data['answer']
        }
        appendMessage(msg, 'incoming')
        scrollToBottom()
    })
    .catch((error) => {
        console.log('Error:', error);
    });


}

function appendMessage(msg, type) {
    let mainDiv = document.createElement('div')
    let className = type
    mainDiv.classList.add(className, 'message')

    let markup = `
        <h4>${msg.user}</h4>
        <p>${msg.message}</p>
    `
    mainDiv.innerHTML = markup
    messageArea.appendChild(mainDiv)
}

// Recieve messages 

// socket.on('message', (msg) => {
//     appendMessage(msg, 'incoming')
//     scrollToBottom()
// })

function scrollToBottom() {
    messageArea.scrollTop = messageArea.scrollHeight
}