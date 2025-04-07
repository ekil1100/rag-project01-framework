// src/pages/Search.jsx
// import React, { useState, useEffect } from 'react'
import { useState, useEffect } from 'react'
// import RandomImage from '../components/RandomImage'
import { apiBaseUrl } from '../config/config'

const Search = () => {
    const [query, setQuery] = useState('')
    const [collection, setCollection] = useState('')
    const [results, setResults] = useState([])
    const [isSearching, setIsSearching] = useState(false)
    const [topK, setTopK] = useState(3)
    const [threshold, setThreshold] = useState(0.7)
    const [collections, setCollections] = useState([])
    const [providers, setProviders] = useState([])
    const [selectedProvider, setSelectedProvider] = useState('milvus')
    const [wordCountThreshold, setWordCountThreshold] = useState(100)
    const [status, setStatus] = useState('')

    // 加载向量数据库providers和collections
    useEffect(() => {
        const fetchData = async () => {
            try {
                // 获取providers列表
                const providersResponse = await fetch(`${apiBaseUrl}/providers`)
                const providersData = await providersResponse.json()
                setProviders(providersData.providers)

                // 获取collections列表 (确保在provider变化时重新获取)
                if (selectedProvider) {
                    const collectionsResponse = await fetch(
                        `${apiBaseUrl}/collections?provider=${selectedProvider}`,
                    )
                    const collectionsData = await collectionsResponse.json()
                    setCollections(collectionsData.collections)
                    setCollection('') // Reset collection choice when provider changes
                    setResults([]) // Clear results when provider changes
                } else {
                    setCollections([]) // Clear collections if no provider selected
                }
            } catch (error) {
                console.error('Error fetching data:', error)
                setCollections([])
                setProviders([])
            }
        }

        fetchData()
    }, [selectedProvider])

    const handleSearch = async () => {
        if (!query || !collection) {
            setStatus('请选择集合并输入搜索内容')
            return
        }

        setIsSearching(true)
        setStatus('')
        setResults([]) // Clear previous results before new search
        try {
            const searchParams = {
                query,
                collection_id: collection,
                top_k: topK,
                threshold,
                word_count_threshold: wordCountThreshold,
            }

            console.log('发送搜索请求:', searchParams)

            const response = await fetch(`${apiBaseUrl}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(searchParams),
            })

            if (!response.ok) {
                const errorData = await response
                    .json()
                    .catch(() => ({ message: 'Unknown error' })) // Try to get error details
                throw new Error(
                    `HTTP error! status: ${response.status} - ${errorData.message || response.statusText}`,
                )
            }

            const data = await response.json()
            console.log('搜索响应:', data) // Log the raw response

            if (
                data &&
                Array.isArray(data.results) &&
                data.results.length > 0
            ) {
                setResults(data.results) // Use data.results directly
                setStatus('搜索完成！')
            } else {
                setResults([])
                if (data && data.message) {
                    setStatus(data.message)
                } else {
                    setStatus('未找到匹配的结果')
                }
            }
        } catch (error) {
            console.error('搜索错误:', error)
            setStatus(`搜索出错: ${error.message}`)
            setResults([])
        } finally {
            setIsSearching(false)
        }
    }

    // 添加保存结果的函数
    const handleSaveResults = async () => {
        if (!results.length) {
            setStatus('没有可保存的搜索结果')
            return
        }

        try {
            const saveParams = {
                query,
                collection_id: collection,
                results: results,
            }

            console.log('发送保存请求:', saveParams)

            const response = await fetch(`${apiBaseUrl}/save-search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(saveParams),
            })

            if (!response.ok) {
                const errorData = await response
                    .json()
                    .catch(() => ({ message: 'Unknown error' }))
                throw new Error(
                    `HTTP error! status: ${response.status} - ${errorData.message || response.statusText}`,
                )
            }

            const data = await response.json()
            setStatus(`结果已保存至: ${data.saved_filepath}`)
        } catch (error) {
            console.error('保存错误:', error)
            setStatus(`保存失败: ${error.message}`)
        }
    }

    return (
        <div className='p-6'>
            <h2 className='text-2xl font-bold mb-6'>Similarity Search</h2>

            <div className='grid grid-cols-12 gap-6'>
                {/* Left Panel - Search Controls */}
                <div className='col-span-3 space-y-4'>
                    <div className='p-4 border rounded-lg bg-white shadow-sm'>
                        <div className='space-y-4'>
                            <div>
                                <label className='block text-sm font-medium mb-1'>
                                    Your Question
                                </label>
                                <textarea
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    placeholder='Enter your search query...'
                                    className='block w-full p-2 border rounded h-32 resize-none'
                                />
                            </div>

                            <div>
                                <label className='block text-sm font-medium mb-1'>
                                    Vector Database
                                </label>
                                <select
                                    value={selectedProvider}
                                    onChange={(e) =>
                                        setSelectedProvider(e.target.value)
                                    }
                                    className='block w-full p-2 border rounded'
                                >
                                    <option value=''>Select Provider...</option>
                                    {providers.map((provider) => (
                                        <option
                                            key={provider.id}
                                            value={provider.id}
                                        >
                                            {provider.name}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label className='block text-sm font-medium mb-1'>
                                    Collection
                                </label>
                                <select
                                    value={collection}
                                    onChange={(e) =>
                                        setCollection(e.target.value)
                                    }
                                    disabled={
                                        !selectedProvider ||
                                        collections.length === 0
                                    }
                                    className='block w-full p-2 border rounded disabled:bg-gray-100'
                                >
                                    <option value=''>
                                        Choose a collection...
                                    </option>
                                    {collections.map((coll) => (
                                        <option key={coll.id} value={coll.id}>
                                            {coll.name} ({coll.count} vectors)
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label className='block text-sm font-medium mb-1'>
                                    Top K Results
                                </label>
                                <input
                                    type='number'
                                    value={topK}
                                    onChange={(e) =>
                                        setTopK(
                                            Math.max(
                                                1,
                                                parseInt(e.target.value) || 1,
                                            ),
                                        )
                                    }
                                    min='1'
                                    max='20'
                                    className='block w-full p-2 border rounded'
                                />
                            </div>

                            <div>
                                <label className='block text-sm font-medium mb-1'>
                                    Similarity Threshold: {threshold.toFixed(2)}
                                </label>
                                <input
                                    type='range'
                                    value={threshold}
                                    onChange={(e) =>
                                        setThreshold(parseFloat(e.target.value))
                                    }
                                    min='0'
                                    max='1'
                                    step='0.05'
                                    className='block w-full'
                                />
                            </div>

                            <div>
                                <label className='block text-sm font-medium mb-1'>
                                    Minimum Word Count: {wordCountThreshold}
                                </label>
                                <input
                                    type='range'
                                    value={wordCountThreshold}
                                    onChange={(e) =>
                                        setWordCountThreshold(
                                            parseInt(e.target.value),
                                        )
                                    }
                                    min='0'
                                    max='500'
                                    step='10'
                                    className='block w-full'
                                />
                            </div>

                            <button
                                onClick={handleSearch}
                                disabled={isSearching || !query || !collection}
                                className='w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed'
                            >
                                {isSearching ? 'Searching...' : 'Search'}
                            </button>
                        </div>
                    </div>

                    {status && (
                        <div
                            className={`p-4 rounded-lg text-sm ${
                                status.includes('错误') ||
                                status.includes('失败')
                                    ? 'bg-red-100 text-red-700'
                                    : status.includes('保存至') ||
                                        status.includes('完成')
                                      ? 'bg-green-100 text-green-700'
                                      : 'bg-blue-100 text-blue-700'
                            }`}
                        >
                            {status}
                        </div>
                    )}
                </div>

                {/* Right Panel - Results */}
                <div className='col-span-9 border rounded-lg bg-white shadow-sm'>
                    {results.length > 0 ? (
                        <div className='p-4'>
                            <div className='flex justify-between items-center mb-4'>
                                <h3 className='text-xl font-semibold'>
                                    Search Results ({results.length})
                                </h3>
                                <button
                                    onClick={handleSaveResults}
                                    className='px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 text-sm'
                                >
                                    保存当前结果
                                </button>
                            </div>
                            <div className='space-y-4 max-h-[calc(100vh-250px)] overflow-y-auto pr-2'>
                                {results.map((result, idx) => (
                                    <div
                                        key={result.id || idx}
                                        className='p-4 border rounded bg-gray-50 hover:bg-gray-100 transition-colors duration-150'
                                    >
                                        <div className='flex justify-between items-start mb-2'>
                                            <span className='font-medium text-sm text-blue-600'>
                                                Score:{' '}
                                                {(result.score * 100).toFixed(
                                                    1,
                                                )}
                                                %
                                            </span>
                                            <div className='text-xs text-gray-500 text-right'>
                                                <div>
                                                    Source:{' '}
                                                    {result.metadata?.source ||
                                                        'N/A'}
                                                </div>
                                                <div>
                                                    Page:{' '}
                                                    {result.metadata?.page ||
                                                        'N/A'}
                                                </div>
                                                <div>
                                                    Chunk ID:{' '}
                                                    {result.metadata?.chunk ??
                                                        'N/A'}
                                                </div>
                                            </div>
                                        </div>
                                        <p className='text-sm whitespace-pre-wrap leading-relaxed'>
                                            {result.text ||
                                                'No content available'}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className='flex items-center justify-center h-full text-gray-500'>
                            {isSearching
                                ? 'Loading results...'
                                : 'Search results will appear here'}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default Search
