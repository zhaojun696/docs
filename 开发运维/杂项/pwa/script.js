




// 监听网络恢复事件
window.addEventListener('online', function() {
    console.log('网络已恢复');
    // 在这里添加你想要在网络恢复时执行的代码
});

// 监听网络断开事件
window.addEventListener('offline', function() {
    console.log('网络已断开');
    // 在这里添加你想要在网络断开时执行的代码
});


function noti() {
    if ('Notification' in window) {
        // 请求通知权限
        Notification.requestPermission().then(function (permission) {
            if (permission === 'granted') {
                // 创建并显示通知
                const notification = new Notification('标题', {
                    body: '这是通知的内容',
                    icon: 'icon.webp', // 可选的图标URL
                    requireInteraction: true
                });

                // 通知的其他操作（可选）
                notification.onclick = function () {
                    window.open('https://baidu.com'); // 点击通知时打开的URL
                };
                // setTimeout(function () {
                //     notification.close();
                // }, 1000); // 1000毫秒 = 1秒

                notification.onclose = () => {
                    console.log('关闭提示')
                }

            } else if (permission === 'denied') {
                console.warn('用户拒绝了通知权限');
            }
        });
    } else {
        console.error('该浏览器不支持Notification API');
    }
}