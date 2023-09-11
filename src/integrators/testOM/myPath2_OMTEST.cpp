#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/renderqueue.h>
#include <iostream>
#include <fstream>
#include "myOM.h"
#include <time.h>
#include <chrono>   
using namespace std;
using namespace chrono;

#if defined(MTS_OPENMP)
#include <omp.h>
#endif

MTS_NAMESPACE_BEGIN

class myPath2OMTESTIntegrator : public Integrator
{
public:
    MTS_DECLARE_CLASS()
    enum SamplingStrategy
    {
        PathBSDF = 0,
        PathNEE,
        PathMIS
    };
    enum MISMode
    {
        UniformHeuristic = 0,
        BalanceHeuristic,
        PowerHeuristic
    };

private:
    int m_maxDepthEye;
    Float m_rrEye;

    ref<Bitmap> m_bitmap;
    std::vector<Point2i> blockOfs;
    int m_blockSize;
    bool m_jitterSample;
    bool m_running;

    std::string m_strategyString;
    SamplingStrategy m_strategy;
    std::string m_MISmodeString;
    MISMode m_MISmode;

    AABB m_baseAABB;
    OM m_om;
    OM* roma;

    double m_connectNum = 0;
    double m_connectTime = 0;

public:
    /// Initialize the integrator with the specified properties
    myPath2OMTESTIntegrator(const Properties &props) : Integrator(props)
    {
        m_maxDepthEye = props.getInteger("maxDepthEye", 50);
        m_rrEye = props.getFloat("rrEye", 0.6);
        m_blockSize = props.getInteger("blockSize", 64);
        m_jitterSample = props.getBoolean("jitterSample", true);
        m_running = true;

        m_strategyString = props.getString("strategy", "mis");
        if (m_strategyString == "bsdf")
            m_strategy = PathBSDF;
        else if (m_strategyString == "nee")
            m_strategy = PathNEE;
        else if (m_strategyString == "mis")
            m_strategy = PathMIS;
        else
            Log(EError, "Unknown strategy: %s", m_strategyString.c_str());

        m_MISmodeString = props.getString("MISmode", "balance");
        if (m_MISmodeString == "uniform")
            m_MISmode = UniformHeuristic;
        else if (m_MISmodeString == "balance")
            m_MISmode = BalanceHeuristic;
        else if (m_MISmodeString == "power")
            m_MISmode = PowerHeuristic;
        else
            Log(EError, "Unknown MIS mode: %s", m_MISmodeString.c_str());
    }

    // Unserialize from a binary data stream
    myPath2OMTESTIntegrator(Stream *stream, InstanceManager *manager)
        : Integrator(stream, manager) {}

    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        Integrator::serialize(stream, manager);
    }

    void printInfos()
    {
        std::ostringstream oss;
        oss << "\n--------- myPath2 Info Print ----------\n";
        oss << "maxDepthEye: " << m_maxDepthEye << endl;
        oss << "rrEye: " << m_rrEye << endl;
        oss << "blockSize: " << m_blockSize << endl;
        oss << "connect number: " << m_connectNum << endl;
        oss <<"time per connect: "<<m_connectTime/m_connectNum*1000<<"ns"<<endl;
        oss << "-----------------------------------------\n";
        SLog(EInfo, oss.str().c_str());
    }

    /// Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue,
                    const RenderJob *job, int sceneResID, int cameraResID,
                    int samplerResID)
    {
        Integrator::preprocess(scene, queue, job, sceneResID,
                               cameraResID, samplerResID);

        blockOfs.clear();
        const Film *film = scene->getSensor()->getFilm();
        Vector2i cropSize = film->getCropSize();
        Point2i cropOffset = film->getCropOffset();
        int w = cropSize.x / m_blockSize + 1;
        int h = cropSize.y / m_blockSize + 1;
        for (int i = 0; i < w; ++i)
        {
            for (int j = 0; j < h; ++j)
            {
                blockOfs.push_back(
                    Point2i(cropOffset.x + i * m_blockSize, cropOffset.y + j * m_blockSize));
            }
        }

        Point m_min(1e30f), m_max(-1e30f), m_center, m_lcorner;
        auto meshes = scene->getMeshes();

        for (auto m : meshes)
        {
            // SLog(EInfo, m->toString().c_str());
            m_min.x = std::min(m_min.x, m->getAABB().min.x);
            m_min.y = std::min(m_min.y, m->getAABB().min.y);
            m_min.z = std::min(m_min.z, m->getAABB().min.z);
            m_max.x = std::max(m_max.x, m->getAABB().max.x);
            m_max.y = std::max(m_max.y, m->getAABB().max.y);
            m_max.z = std::max(m_max.z, m->getAABB().max.z);
        }

        /* generate roma */
        Vector3 d = m_max - m_min;
        Float r = d.length();
        r *= 0.5 * 1.001f;
        m_center = m_min + d / 2.0f;
        m_lcorner = m_center - Vector3(r);
        m_baseAABB = AABB(m_lcorner, m_lcorner + Vector3(2 * r));

        m_om.clear();
        m_om.setAABB(m_baseAABB);
        // m_om.setSize(2 * r);
        m_om.setScene(scene);

        /* init roma */
        roma = new OM[OMNUM];
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < OMNUMSQRT; i++)
            for (int j = 0; j < OMNUMSQRT; j++)
            {
                roma[i * OMNUMSQRT + j].clear();
                roma[i * OMNUMSQRT + j].setAABB(m_baseAABB);
                m_om.generateROMA(&roma[i * OMNUMSQRT + j], Point2((i + 0.5f) / OMNUMSQRT, (j + 0.5f) / OMNUMSQRT));
            }

        printInfos();
        return true;
    }

    void cancel()
    {
        m_running = false;
    }

    bool render(Scene *scene, RenderQueue *queue,
                const RenderJob *job, int sceneResID, int sensorResID, int unused)
    {

        ref<Scheduler> sched = Scheduler::getInstance();
        int blockSize = scene->getBlockSize();

        ref<Sampler> sampler = scene->getSampler();
        size_t sampleCount = sampler->getSampleCount();

        ref<Sensor> sensor = scene->getSensor();
        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        ref<Film> film = sensor->getFilm();
        Vector2i cropSize = film->getCropSize();
        Point2i cropOffset = film->getCropOffset();

        /* Allocate memory */
        m_bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, cropSize);
        m_bitmap->clear();

        Spectrum *target = (Spectrum *)m_bitmap->getUInt8Data();
        // Warm up
        for (int i = 0; i < 1; ++i) {
            SLog(EInfo, "Frame: %i", i);
            if (!m_running)
                break;
            int blockCnt = (int)blockOfs.size();
            for (int block = 0; block < blockCnt; ++block) {
                Point2i &bOfs = blockOfs[block];
                int xBlockOfs = bOfs.x;
                int yBlockOfs = bOfs.y;
                for (int yofs = 0; yofs < m_blockSize; ++yofs) {
                    for (int xofs = 0; xofs < m_blockSize; ++xofs) {
                        int xRealOfs = xBlockOfs + xofs;
                        int yRealOfs = yBlockOfs + yofs;
                        if (xRealOfs >= cropSize.x || yRealOfs >= cropSize.y)
                            continue;
                        Point2 apertureSample, samplePos;
                        samplePos = Point2(0.5, 0.5);
                        if (m_jitterSample)
                            samplePos = sampler->next2D();
                        samplePos.y += cropOffset.y + yRealOfs;
                        samplePos.x += cropOffset.x + xRealOfs;
                        RayDifferential eyeRay;
                        Spectrum sampleValue = sensor->sampleRay(
                            eyeRay, samplePos, apertureSample, 0.0f);
                        int ofs = yRealOfs * cropSize.x + xRealOfs;
                        Spectrum L = Li(eyeRay, scene, sampler, 0, 0);
                    }
                }
            }
        }
         
        
        auto start = high_resolution_clock::now();
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);

        
        double num1 = 0;
        start = high_resolution_clock::now();
        for (int i = 0; i < sampleCount; ++i) {
            SLog(EInfo, "Frame: %i", i);
            if (!m_running)
                break;
            /* Trace eye subpath*/
            int blockCnt = (int)blockOfs.size();

            for (int block = 0; block < blockCnt; ++block)
            {
                Point2i &bOfs = blockOfs[block];
                int xBlockOfs = bOfs.x;
                int yBlockOfs = bOfs.y;
                for (int yofs = 0; yofs < m_blockSize; ++yofs)
                {
                    for (int xofs = 0; xofs < m_blockSize; ++xofs)
                    {
                        int xRealOfs = xBlockOfs + xofs;
                        int yRealOfs = yBlockOfs + yofs;
                        if (xRealOfs >= cropSize.x || yRealOfs >= cropSize.y)
                            continue;
                        Point2 apertureSample, samplePos;
                        samplePos = Point2(0.5, 0.5);
                        if (m_jitterSample)
                            samplePos = sampler->next2D();
                        samplePos.y += cropOffset.y + yRealOfs;
                        samplePos.x += cropOffset.x + xRealOfs;

                        RayDifferential eyeRay;
                        Spectrum sampleValue = sensor->sampleRay(
                            eyeRay, samplePos, apertureSample, 0.0f);
                        int ofs = yRealOfs * cropSize.x + xRealOfs;
                        Spectrum L = Li(eyeRay, scene, sampler, &num1, 0);
                    }
                }
            }
        }
        end   = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end - start);
        double time1 = double(duration.count());  
        
        start = high_resolution_clock::now();
        double num2 = 0;
        for (int i = 0; i < sampleCount; ++i)
        {
            SLog(EInfo, "Frame: %i", i);
            if (!m_running)
                break;
            int blockCnt = (int)blockOfs.size();
            for (int block = 0; block < blockCnt; ++block)
            {
                Point2i &bOfs = blockOfs[block];
                int xBlockOfs = bOfs.x;
                int yBlockOfs = bOfs.y;
                for (int yofs = 0; yofs < m_blockSize; ++yofs)
                {
                    for (int xofs = 0; xofs < m_blockSize; ++xofs)
                    {
                        int xRealOfs = xBlockOfs + xofs;
                        int yRealOfs = yBlockOfs + yofs;
                        if (xRealOfs >= cropSize.x || yRealOfs >= cropSize.y)
                            continue;
                        Point2 apertureSample, samplePos;
                        samplePos = Point2(0.5, 0.5);
                        if (m_jitterSample)
                            samplePos = sampler->next2D();
                        samplePos.y += cropOffset.y + yRealOfs;
                        samplePos.x += cropOffset.x + xRealOfs;

                        RayDifferential eyeRay;
                        Spectrum sampleValue = sensor->sampleRay(
                            eyeRay, samplePos, apertureSample, 0.0f);
                        int ofs = yRealOfs * cropSize.x + xRealOfs;
                        Spectrum L = Li(eyeRay, scene, sampler, &num2, 100);
                    }
                }
            }
        }
        end   = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end - start);
        double time2 = double(duration.count());  
        

        m_connectTime += time2 - time1;
        m_connectNum += num2 - num1;   
        printInfos();
        delete[] roma;
        return true;
    }

    inline Float misWeight(Float pdfBSDF, Float pdfDirect, SamplingStrategy strategy) const
    {
        switch (strategy)
        {
        case PathBSDF:
            switch (m_strategy)
            {
            case PathBSDF:
                return 1;
            case PathNEE:
                return 0;
            case PathMIS:
                return mis(pdfBSDF, pdfDirect, m_MISmode);
            default:
                return 0;
            }
        case PathNEE:
            switch (m_strategy)
            {
            case PathBSDF:
                return 0;
            case PathNEE:
                return 1;
            case PathMIS:
                return mis(pdfDirect, pdfBSDF, m_MISmode);
            default:
                return 0;
            }
        default:
            return 0;
        }
    }

#define sqr(x) ((x) * (x))
    inline Float mis(Float p1, Float p2, MISMode mode) const
    {
        switch (m_MISmode)
        {
        case UniformHeuristic:
            return 0.5;
        case BalanceHeuristic:
            return p1 / (p1 + p2);
        case PowerHeuristic:
            return sqr(p1) / (sqr(p1) + sqr(p2));
        default:
            return 0;
        }
    }

    /// Query for an unbiased estimate of the radiance along <tt>r</tt>
    Spectrum Li(const RayDifferential &r, Scene *scene, Sampler *sampler, double *cNum, int shadowTest)
    {
        /* Some aliases and local variables */
        Intersection its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        scene->rayIntersect(ray, its);
        ray.mint = Epsilon;

        if (its.isValid() && its.isEmitter())
            return its.Le(-ray.d);

        int depth = 1;
        while (depth <= m_maxDepthEye)
        {
            if (!its.isValid())
                break;

            const BSDF *bsdf = its.getBSDF(ray);
            DirectSamplingRecord dRec(its);

            if (bsdf->getType() & BSDF::ESmooth)
            {
                Spectrum value = scene->sampleEmitterDirect(dRec, sampler->next2D(), false);

                int id = OM::nearestOMindex(dRec.d);
                if (id < 0 || id >= OMNUM)
                {
                    SLog(EError, "id error: %d\n", id);
                }
                bool vis = false;
                if(depth == -10)
                {
                    value = scene->sampleEmitterDirect(dRec, sampler->next2D());
                    vis = true;
                } else {
                    // bool vis = roma[id].visibilityBOM(its.p + its.shFrame.n * 0.5, dRec.p);
                    bool vis = roma[id].Visible(its.p + its.shFrame.n * 0.5, dRec.p);
                    for (int i = 0; i < shadowTest; i++){
                        int id = OM::nearestOMindex(dRec.d);
                        //vis |= roma[id].Visible(its.p + its.shFrame.n * (0.5+0.001*i), dRec.p);
                    }
                    if(cNum)
                        *cNum += shadowTest + 1.0f;
                }
                if (vis && !value.isZero())
                {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    if (!bsdfVal.isZero())
                    {
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                                            ? bsdf->pdf(bRec)
                                            : 0;
                        Float misW = misWeight(bsdfPdf, dRec.pdf, PathNEE);
                        Li += throughput * value * bsdfVal * misW;
                    }
                }
            }

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, sampler->next2D());
            if (bsdfWeight.isZero())
                break;

            const Vector wo = its.toWorld(bRec.wo);

            throughput *= bsdfWeight;
            eta *= bRec.eta;

            ray = Ray(its.p, wo, ray.time);

            Spectrum value;
            if (scene->rayIntersect(ray, its))
            {
                if (its.isEmitter())
                {
                    value = its.Le(-ray.d);
                    dRec.setQuery(ray, its);
                    Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec) : 0;
                    Float misW = misWeight(bsdfPdf, lumPdf, PathBSDF);
                    Li += throughput * value * misW;
                    return Li;
                }
            }

            Float q = std::min(throughput.max() * eta * eta, (Float)0.95f);
            if (sampler->next1D() >= q)
                break;
            throughput /= q;
            ++depth;
        }
        return Li;
    }
};

MTS_IMPLEMENT_CLASS_S(myPath2OMTESTIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(myPath2OMTESTIntegrator, "myPath2_OMTEST");
MTS_NAMESPACE_END