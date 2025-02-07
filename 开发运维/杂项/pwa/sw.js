const CACHE_NAME = 'pwa-example-cache-v3';
const urlsToCache = [
    '/',
    '/index.html',
    '/style.css',
    '/script.js',
    '/icon.webp'
];

const maps = {
    'i': 0
}

self.addEventListener('sync', event => {
    console.log('收到异步任务',event.tag)
    if (event.tag === 'my-sync-tag') {
        if (maps['i'] > 0) {
            console.log('异步任务以存在',event.tag)
            return
        }else{
            console.log('执行异步任务',event.tag)
        }
        // 执行你的任务
        event.waitUntil(new Promise(() => {
            setInterval(() => {
                const options = {
                    body: 'This is the body of the notification',
                    icon: 'icon.webp',
                    badge: 'icon.webp',
                    data: {
                        url: 'https://baidu.com'
                    }
                };
                maps['i']=maps['i']+1
                console.log('检测全局变量值', maps['i'])
                self.registration.showNotification('Notification Title', options)
                
            }, 1000 * 30)
        }));
    }else if (event.tag === 'my-sync-tag1') {
        console.log('执行异步任务',event.tag)
    }
})
self.addEventListener('notificationclick', function(event) {
    console.log('收到notificationclick',event.notification.data)
    event.notification.close(); // Close the notification
    event.waitUntil(
        event.waitUntil(
            clients.matchAll({ type: 'window' }).then(function(clientList) {
                // 检查是否有已打开的客户端
                const client = clientList.find(c => c.url === event.notification.data.url && 'focus' in c);
                if (client) {
                    return client.focus(); // 激活已打开的客户端
                } else {
                    return clients.openWindow(event.notification.data.url); // 否则打开新窗口
                }
            })
        )
    )
})

self.addEventListener('periodicsync', event => {
    console.log('收到周期性异步任务',event.tag)
    if (event.tag === 'daily-sync') {

    }
})
  
self.addEventListener('install', event => {
    console.log('触发安装')

    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('Opened cache');
                return cache.addAll(urlsToCache);
            })
    );
});




self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                console.log(response, event.request)
                if (response) {
                    return response;
                }
                if (event.request.url.endsWith('.js')) {
                    // 克隆请求，因为请求是一个流，只能使用一次
                    const fetchRequest = event.request.clone();
                    return fetch(fetchRequest)
                        .then(response => {
                            // 检查响应是否有效
                            if (!response || response.status !== 200) {
                                return response;
                            }

                            // 克隆响应，因为响应是一个流，只能使用一次
                            const responseToCache = response.clone();

                            caches.open(CACHE_NAME)
                                .then(cache => {
                                    cache.put(event.request, responseToCache);
                                });

                            return response;
                        });
                }
                return fetch(event.request);
            })
    );
});






self.addEventListener('message', function (event) {
    console.log('收到消息:', event.data);

    event.source.postMessage(event.data)
});

self.addEventListener('activate', event => {
    console.log('激活新的 Service Worker');
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('删除旧的缓存', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});


// 1. 修改 sw.js 文件
// 首先，你需要修改 sw.js 文件。这可能包括添加新的缓存策略、更新事件处理程序或其他逻辑。

// 2. 浏览器检测更新
// 当浏览器访问包含 Service Worker 的页面时，它会自动检查 sw.js 文件是否有更新。浏览器会比较当前 sw.js 文件的内容与缓存中的版本，如果发现差异，就会下载新的 sw.js 文件。

// 3. 安装新的 Service Worker
// 一旦下载了新的 sw.js 文件，浏览器会尝试安装新的 Service Worker。这个过程会触发 install 事件。

// 4. 激活新的 Service Worker
// 如果新的 Service Worker 成功安装，它会进入 waiting 状态，等待当前活动的 Service Worker 控制的页面全部关闭。一旦所有页面都关闭，新的 Service Worker 就会激活，并触发 activate 事件。

// 5. 清理旧的缓存
// 在 activate 事件中，你可以清理旧的缓存。例如，删除不再需要的缓存版本。

// 6. 强制更新
// 如果你希望用户在访问页面时立即使用新的 Service Worker，而不是等待当前页面关闭，可以在注册 Service Worker 时使用 skipWaiting 方法。
// if (registration.waiting) {
//     registration.waiting.postMessage('skipWaiting');
// }
// self.addEventListener('message', event => {
//     if (event.data === 'skipWaiting') {
//         self.skipWaiting();
//     }
// });
