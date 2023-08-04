#include <mitsuba/render/scene.h>
#include "myOM.h"

MTS_NAMESPACE_BEGIN
class myPathIntegrator : public SamplingIntegrator {
public:
    MTS_DECLARE_CLASS()
    enum SamplingStrategy {
        PathBSDF = 0,
        PathNEE,
        PathMIS
    };
    enum MISMode {
        UniformHeuristic = 0,
        BalanceHeuristic,
        PowerHeuristic
    };

private:
    int m_maxDepth;
    int m_rrDepth;
    static int m_LiCount;
    std::string m_strategyString;
    SamplingStrategy m_strategy;
    std::string m_MISmodeString;
    MISMode m_MISmode;

    AABB m_baseAABB;
    OM m_om;
    OM roma[OMNUM];

public:
    /// Initialize the integrator with the specified properties
    myPathIntegrator(const Properties &props) : SamplingIntegrator(props) {
        m_maxDepth = props.getInteger("maxDepth", 50);
        m_rrDepth = props.getInteger("rrDepth", 0);

        m_strategyString = props.getString("strategy", "nee");
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
    myPathIntegrator(Stream *stream, InstanceManager *manager)
        : SamplingIntegrator(stream, manager) {}

    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const {
        SamplingIntegrator::serialize(stream, manager);
    }

    void generateROMA(Sampler* sampler) {
        for (int i = 0; i < OMNUMSQRT; i++)
            for (int j = 0; j < OMNUMSQRT; j++) {
                roma[i * OMNUMSQRT + j].clear();
                roma[i * OMNUMSQRT + j].setAABB(m_baseAABB);
                Point2 uv = sampler->next2D();
                m_om.generateROMA(&roma[i * OMNUMSQRT + j], Point2((i + uv.x) / OMNUMSQRT, (j + uv.y) / OMNUMSQRT));
            }
    }

    /// Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue,
                    const RenderJob *job, int sceneResID, int cameraResID,
                    int samplerResID)
    {
        SamplingIntegrator::preprocess(scene, queue, job, sceneResID,
                                       cameraResID, samplerResID);

        Point m_min(1e30f), m_max(-1e30f), m_center, m_lcorner;
        auto meshes = scene->getMeshes();

        for (auto m : meshes) {
            // SLog(EInfo, m->toString().c_str());
            m_min.x = std::min(m_min.x, m->getAABB().min.x);
            m_min.y = std::min(m_min.y, m->getAABB().min.y);
            m_min.z = std::min(m_min.z, m->getAABB().min.z);
            m_max.x = std::max(m_max.x, m->getAABB().max.x);
            m_max.y = std::max(m_max.y, m->getAABB().max.y);
            m_max.z = std::max(m_max.z, m->getAABB().max.z);
        }

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
        for (int i = 0; i < OMNUMSQRT; i++)
            for (int j = 0; j < OMNUMSQRT; j++) {
                roma[i * OMNUMSQRT + j].clear();
                roma[i * OMNUMSQRT + j].setAABB(m_baseAABB);
                m_om.generateROMA(&roma[i * OMNUMSQRT + j], Point2((i + 0.5f) / OMNUMSQRT, (j + 0.5f) / OMNUMSQRT));
            }

        printInfos();
    
        return true;
    }

    inline Float misWeight(Float pdfBSDF, Float pdfDirect, SamplingStrategy strategy) const {
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
    inline Float mis(Float p1, Float p2, MISMode mode) const {
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
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        // generateROMA(rRec.sampler);
        ++m_LiCount;
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        if (its.isValid() && its.isEmitter())
            return its.Le(-ray.d);

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid())
                break;

            const BSDF *bsdf = its.getBSDF(ray);

            /* Estimate the direct illumination if this is requested */
            DirectSamplingRecord dRec(its);

            if (m_strategy != PathBSDF && (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D(), false);
                
                int id = OM::nearestOMindex(dRec.d);
                if (id <0 || id >= OMNUM) {
                    SLog(EError, "id error: %d\n", id);
                }

                bool vis = roma[id].Visible(its.p+its.shFrame.n * 0.5, dRec.p);
                // bool vis = roma[id].visibilityBOM(its.p+its.shFrame.n * 0.5, dRec.p);
                
                // bool vis = m_om.visibilityBOM(its.p+its.shFrame.n*0.5, dRec.p);

                if (vis) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    if (!bsdfVal.isZero()) {
                        /* Calculate prob. of having generated that direction
                           using BSDF sampling */
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                                            ? bsdf->pdf(bRec)
                                            : 0;
                        /* Weight using the power heuristic */
                        Float misW = misWeight(bsdfPdf, dRec.pdf, PathNEE);
                        Li += throughput * value * bsdfVal * misW;
                    }
                }
            }

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            if (bsdfWeight.isZero())
                break;

            const Vector wo = its.toWorld(bRec.wo);

            /* Keep track of the throughput and relative
               refractive index along the path */
            throughput *= bsdfWeight;
            eta *= bRec.eta;

            /* Trace a ray in this direction */
            ray = Ray(its.p, wo, ray.time);

            Spectrum value;
            if (scene->rayIntersect(ray, its)) {
                if (its.isEmitter()) {
                    value = its.Le(-ray.d);
                    dRec.setQuery(ray, its);
                    Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec) : 0;
                    Float misW = misWeight(bsdfPdf, lumPdf, PathBSDF);
                    Li += throughput * value * misW;
                    return Li;
                }
            }

            if (rRec.depth >= m_rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                   while accounting for the solid angle compression at refractive
                   index boundaries. Stop with at least some probability to avoid
                   getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float)0.95f);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
            }

            ++rRec.depth;
        }
        return Li;
    }

    void printInfos(){
        std::ostringstream oss;
        oss<<"\n--------- PATH-OM Info Print ----------\n";
        oss<<"ROMA size = " <<1.0f*OMNUM*OMSIZE*OMSIZE*OMSIZE/8/1024/1024 << " MB\n";
        oss<<"-----------------------------------------\n";
        SLog(EDebug, oss.str().c_str());
    }
};

int myPathIntegrator::m_LiCount = 0;

MTS_IMPLEMENT_CLASS_S(myPathIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(myPathIntegrator, "My path integrator");
MTS_NAMESPACE_END