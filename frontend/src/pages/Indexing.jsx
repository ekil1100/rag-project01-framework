// src/pages/Indexing.jsx
import { useState, useEffect } from 'react'
import RandomImage from '../components/RandomImage'
import { apiBaseUrl } from '../config/config'

const Indexing = () => {
    const [embeddingFile, setEmbeddingFile] = useState('')
    const [vectorDb, setVectorDb] = useState('chroma')
    const [indexMode, setIndexMode] = useState('standard')
    const [status, setStatus] = useState('')
    const [embeddedFiles, setEmbeddedFiles] = useState([])
    const [indexingResult, setIndexingResult] = useState(null)
    const [collections, setCollections] = useState([])
    const [selectedCollection, setSelectedCollection] = useState('')
    const [providers, setProviders] = useState([])

    // 数据库和索引模式的配置
    const dbConfigs = {
        pinecone: {
            modes: ['standard', 'hybrid'],
        },
        milvus: {
            modes: ['flat', 'ivf_flat', 'ivf_sq8', 'hnsw'],
        },
        qdrant: {
            modes: ['hnsw', 'custom'],
        },
        weaviate: {
            modes: ['hnsw', 'flat'],
        },
        chroma: {
            modes: ['hnsw', 'standard'],
        },
        faiss: {
            modes: ['flat', 'ivf', 'hnsw'],
        },
    }

    useEffect(() => {
        fetchEmbeddedFiles()
        // fetchCollections is now called inside the other useEffect
    }, [])

    useEffect(() => {
        // 当数据库改变时，重置索引模式为该数据库的第一个可用模式
        if (dbConfigs[vectorDb]) {
            setIndexMode(dbConfigs[vectorDb].modes[0])
        } else {
            // Handle case where vectorDb might not be in dbConfigs initially or after an update
            console.warn(`Configuration for ${vectorDb} not found.`)
            // Optionally set a default index mode or handle error
        }

        // Fetch providers and collections when vectorDb changes or on initial load
        const fetchData = async () => {
            try {
                // 获取providers列表 (only needed once, could be moved to the initial useEffect)
                if (providers.length === 0) {
                    const providersResponse = await fetch(
                        `${apiBaseUrl}/providers`,
                    )
                    const providersData = await providersResponse.json()
                    setProviders(providersData.providers)

                    // If initial vectorDb is not in the fetched providers, update it?
                    // Or just ensure the default 'milvus' exists or handle gracefully.
                    if (
                        !providersData.providers.some((p) => p.id === vectorDb)
                    ) {
                        // console.warn(`Default DB ${vectorDb} not in fetched providers.`);
                        // Optionally set vectorDb to the first available provider
                        if (providersData.providers.length > 0) {
                            // setVectorDb(providersData.providers[0].id); // This would trigger another re-render/fetch cycle
                        }
                    }
                }

                // 获取collections列表 based on the currently selected vectorDb
                const collectionsResponse = await fetch(
                    `${apiBaseUrl}/collections?provider=${vectorDb}`, // Use vectorDb here
                )
                const collectionsData = await collectionsResponse.json()
                setCollections(collectionsData.collections || []) // Ensure collections is always an array
                setSelectedCollection('') // Reset selected collection when DB changes
                setIndexingResult(null) // Clear results when DB changes
            } catch (error) {
                console.error('Error fetching data:', error)
                setStatus('Error fetching provider/collection data.') // Update status
            }
        }

        fetchData()
    }, [vectorDb]) // Depend on vectorDb

    const fetchEmbeddedFiles = async () => {
        try {
            const response = await fetch(`${apiBaseUrl}/list-embedded`)
            const data = await response.json()
            if (data.documents) {
                setEmbeddedFiles(
                    data.documents.map((doc) => ({
                        ...doc,
                        id: doc.name,
                        displayName: doc.name,
                    })),
                )
            }
        } catch (error) {
            console.error('Error fetching embedded files:', error)
            setStatus('Error loading embedding files')
        }
    }

    const handleIndex = async () => {
        if (!embeddingFile) {
            setStatus('Please select an embedding file')
            return
        }

        setStatus('Indexing...')
        try {
            const response = await fetch(`${apiBaseUrl}/index`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    fileId: embeddingFile,
                    vectorDb,
                    indexMode,
                }),
            })

            const data = await response.json()
            setIndexingResult(data)
            setStatus('Indexing completed successfully')
        } catch (error) {
            console.error('Error indexing:', error)
            setStatus('Error during indexing: ' + error.message)
        }
    }

    const handleDisplay = async (collectionName) => {
        if (!collectionName) return

        try {
            // Use vectorDb instead of selectedProvider
            const response = await fetch(
                `${apiBaseUrl}/collections/${vectorDb}/${collectionName}`,
            )
            const data = await response.json()

            // 只包含有实际值的属性
            const result = {
                database: vectorDb, // Use vectorDb
                collection_name: data.name,
                // Use optional chaining and nullish coalescing for safety
                total_vectors: data.num_entities ?? data.vectors_count ?? 'N/A',
                index_size: data.num_entities ?? data.vectors_count ?? 'N/A',
            }

            // 只在有实际值时添加可选属性
            const vectorField = data.schema?.fields?.find(
                (f) =>
                    f.is_primary !== true &&
                    (f.data_type === 'FloatVector' || f.type === 'vector'),
            ) // Adjust based on actual schema structure across DBs
            const indexParams = vectorField?.index_params ?? data.index_params // Fallback if index_params are top-level
            const indexType = indexParams?.index_type ?? data.index_type // Further fallback

            if (indexType) {
                result.index_mode = indexType
            } else if (data.hnsw_config || data.quantization_config) {
                // Attempt to infer index type for specific DBs like Qdrant if not directly provided
                if (data.hnsw_config) result.index_mode = 'HNSW'
                // Add more specific checks if needed
            }

            if (data.processing_time) {
                result.processing_time = data.processing_time
            }

            // Add other relevant fields if available, e.g., dimensions
            const dimension =
                vectorField?.params?.dim ??
                data.vector_params?.size ??
                data.schema?.fields?.find((f) => f.name === 'vector')?.params
                    ?.dim
            if (dimension) {
                result.dimensions = dimension
            }

            setIndexingResult(result)
            setStatus(`Displayed collection: ${collectionName}`) // Update status
        } catch (error) {
            console.error('Error displaying collection:', error)
            setStatus(
                `Error displaying collection ${collectionName}: ${error.message}`,
            ) // Update status
        }
    }

    const handleDelete = async (collectionName) => {
        if (!collectionName) return

        if (
            window.confirm(
                `Are you sure you want to delete collection "${collectionName}" from ${vectorDb}?`, // Specify DB
            )
        ) {
            try {
                // Use vectorDb instead of selectedProvider
                await fetch(
                    `${apiBaseUrl}/collections/${vectorDb}/${collectionName}`,
                    {
                        method: 'DELETE',
                    },
                )
                setSelectedCollection('')
                setIndexingResult(null) // Clear results after deletion
                setStatus(
                    `Collection "${collectionName}" deleted successfully.`,
                ) // Update status
                // Re-fetch collections for the current DB
                const response = await fetch(
                    `${apiBaseUrl}/collections?provider=${vectorDb}`, // Use vectorDb
                )
                const data = await response.json()
                setCollections(data.collections || []) // Ensure it's an array
            } catch (error) {
                console.error('Error deleting collection:', error)
                setStatus(
                    `Error deleting collection ${collectionName}: ${error.message}`,
                ) // Update status
            }
        }
    }

    return (
        <div className='p-6'>
            <h2 className='text-2xl font-bold mb-6'>
                Vector Database Indexing
            </h2>

            <div className='grid grid-cols-12 gap-6'>
                {/* Left Panel - Controls */}
                <div className='col-span-3'>
                    <div className='p-4 border rounded-lg bg-white shadow-sm space-y-4'>
                        {/* Embedding File Selection */}
                        <div>
                            <label className='block text-sm font-medium mb-1'>
                                Embedding File
                            </label>
                            <select
                                value={embeddingFile}
                                onChange={(e) =>
                                    setEmbeddingFile(e.target.value)
                                }
                                className='block w-full p-2 border rounded'
                            >
                                <option value=''>Choose a file...</option>
                                {embeddedFiles.map((file) => (
                                    <option key={file.name} value={file.name}>
                                        {file.displayName}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Vector Database Selection */}
                        <div>
                            <label className='block text-sm font-medium mb-1'>
                                Vector Database
                            </label>
                            <select
                                value={vectorDb}
                                onChange={(e) => setVectorDb(e.target.value)}
                                className='block w-full p-2 border rounded'
                            >
                                {Object.keys(dbConfigs).map((provider) => (
                                    <option key={provider} value={provider}>
                                        {provider.toUpperCase()}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Index Mode Selection */}
                        <div>
                            <label className='block text-sm font-medium mb-1'>
                                Index Mode
                            </label>
                            <select
                                value={indexMode}
                                onChange={(e) => setIndexMode(e.target.value)}
                                className='block w-full p-2 border rounded'
                            >
                                {dbConfigs[vectorDb]?.modes.map((mode) => (
                                    <option key={mode} value={mode}>
                                        {mode.toUpperCase()}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Action Buttons and Collection Management */}
                        <div className='space-y-2'>
                            {/* Index Data Button */}
                            <button
                                onClick={handleIndex}
                                className='w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300'
                                disabled={!embeddingFile}
                            >
                                Index Data
                            </button>

                            {/* Collection Selection */}
                            <div>
                                <label className='block text-sm font-medium mb-1'>
                                    Collection
                                </label>
                                <select
                                    value={selectedCollection}
                                    onChange={(e) =>
                                        setSelectedCollection(e.target.value)
                                    }
                                    className='block w-full p-2 border rounded'
                                >
                                    <option value=''>
                                        Choose a collection...
                                    </option>
                                    {collections.map((coll) => (
                                        <option key={coll.id} value={coll.id}>
                                            {coll.name} ({coll.count} documents)
                                        </option>
                                    ))}
                                </select>
                            </div>

                            {/* Display Collection Button */}
                            <button
                                onClick={() =>
                                    handleDisplay(selectedCollection)
                                }
                                disabled={!selectedCollection}
                                className='w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300'
                            >
                                Display Collection
                            </button>

                            {/* Delete Collection Button */}
                            <button
                                onClick={() => handleDelete(selectedCollection)}
                                disabled={!selectedCollection}
                                className='w-full px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-red-300'
                            >
                                Delete Collection
                            </button>
                        </div>

                        {status && (
                            <div className='mt-4 p-3 rounded border bg-gray-50'>
                                <p className='text-sm'>{status}</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Right Panel - Results */}
                <div className='col-span-9 border rounded-lg bg-white shadow-sm'>
                    {indexingResult ? (
                        <div className='p-4'>
                            <h3 className='text-xl font-semibold mb-4'>
                                Indexing Results
                            </h3>
                            <div className='space-y-3'>
                                <div className='p-3 border rounded bg-gray-50'>
                                    <div className='text-sm text-gray-600'>
                                        <p>
                                            Database: {indexingResult.database}
                                        </p>
                                        {indexingResult.index_mode && (
                                            <p>
                                                Index Mode:{' '}
                                                {indexingResult.index_mode}
                                            </p>
                                        )}
                                        <p>
                                            Total Vectors:{' '}
                                            {indexingResult.total_vectors}
                                        </p>
                                        <p>
                                            Index Size:{' '}
                                            {indexingResult.index_size}
                                        </p>
                                        {indexingResult.processing_time && (
                                            <p>
                                                Processing Time:{' '}
                                                {indexingResult.processing_time}
                                                s
                                            </p>
                                        )}
                                        <p>
                                            Collection Name:{' '}
                                            {indexingResult.collection_name}
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <RandomImage message='Indexing results will appear here' />
                    )}
                </div>
            </div>
        </div>
    )
}

export default Indexing
