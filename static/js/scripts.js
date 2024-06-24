$(document).ready(function() {
    $('#chat-form').on('submit', function(event) {
        event.preventDefault();
        const userQuestion = $('#question').val();

        $.ajax({
            type: 'POST',
            url: '/ask',
            contentType: 'application/json',
            data: JSON.stringify({ question: userQuestion }),
            success: function(response) {
                const chatHistory = response.chat_history;
                updateChatBox(chatHistory);
                $('#question').val('');
            }
        });
    });

    function updateChatBox(chatHistory) {
        const chatBox = $('#chat-box');
        chatBox.empty();

        chatHistory.forEach(message => {
            const messageClass = message.role === 'user' ? 'user' : 'bot';
            const messageDiv = `<div class="message ${messageClass}">${message.content}</div>`;
            chatBox.append(messageDiv);
        });

        chatBox.scrollTop(chatBox[0].scrollHeight);
    }
});
